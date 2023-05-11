import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import time
import argparse
from pathlib import Path
import librosa
import soundfile as sf
import subprocess
from d2m.dataset import S25Dataset
from d2m.loris_modules import LORIS
from d2m.utils import save_sample, seed_everything, load_yaml_config, get_model_parameters_info, instantiate_from_config
from d2m.engine.distributed import set_dist_rank, distribute_model
from d2m.engine.logger import Logger
from d2m.engine.lr_scheduler import ReduceLROnPlateauWithWarmup
import warnings
import sys
warnings.simplefilter("ignore")
os.environ["KMP_WARNINGS"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='./configs/')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--num_node', type=int, default=1, help='number of nodes for distributed training')
    parser.add_argument('--node_rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', type=int, default=0, help="distribted training")
    parser.add_argument('--local_rank', type=int, default=0, help="distribted training")
    parser.add_argument('--world_size', type=int, default=0, help="distribted training")
    parser.add_argument('--seed', type=int, default=2333, help='seed for initializing training. ')
    parser.add_argument('--cudnn_deterministic', action='store_true', help='set cudnn.deterministic True')
    parser.add_argument('--tensorboard', type=bool, default=False, help='use tensorboard for logging')
    args = parser.parse_args()
    return args

def train(config, args, logger):
    log_path = config['log_path']
    model_save_path = config['model_save_path']
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    batch_size = config['batch_size']
    if args.tensorboard is True:
        writer = SummaryWriter(str(log_path))

    loris = LORIS(config['model'])
    # load pre-trained music diffusion checkpoints
    autoencoder_state_dict = torch.load(config['autoencoder_path'])
    autoencoder_type = config['model']['autoencoder_type']

    if config['model']['use_pretrain'] is True:
        if autoencoder_type == 'autoencoder':
            autoencoder_state_dict = {k: v for k, v in autoencoder_state_dict.items() if loris.autoencoder.state_dict()[k].numel() == v.numel()}
            loris.autoencoder.load_state_dict(autoencoder_state_dict, strict=False)
        elif autoencoder_type == 'diffusion':
            loris.diffusion.load_state_dict(autoencoder_state_dict, strict=True)
        elif autoencoder_type == 'cond_diffusion':
            loris.cond_diffusion.load_state_dict(autoencoder_state_dict, strict=False)
        else:
            raise NotImplementedError ("Unrecognised Autoencoder Type!")

    #### create optimizer #####
    b_params = []   # backbone params
    a_params = []   # additional params
    for k, v in loris.cond_diffusion.named_parameters():
        if k in autoencoder_state_dict.keys() and autoencoder_state_dict[k].numel() == v.numel():
            b_params.append(v)
        else:
            a_params.append(v)
    params = [
        {"params": a_params, "lr": config['base_lr']},
        {"params": b_params, "lr": config['backbone_base_lr']},
    ]
    logger.log_info(str(get_model_parameters_info(loris)))
    opt = torch.optim.AdamW(params, betas=config['betas'], weight_decay=config['weight_decay'])
    logger.log_info("Finish creating the optimizer.")
    #### creat data loader ####
    train_dataset = S25Dataset(audio_files=config['audio_train_path'], video_files=config['video_train_path'], motion_files=config['motion_train_path'], genre_label=config['genre_train_path'], augment=True, config=config['model'])
    val_dataset = S25Dataset(audio_files=config['audio_test_path'], video_files=config['video_test_path'], motion_files=config['motion_test_path'], genre_label=config['genre_test_path'], augment=False, config=config['model'])
    scheduler = ReduceLROnPlateauWithWarmup(opt, factor=config['lr_scheduler']['factor'], patience=config['lr_scheduler']['patience'],
                threshold=config['lr_scheduler']['threshold'], threshold_mode=config['lr_scheduler']['threshold_mode'],
                min_lr=config['lr_scheduler']['min_lr'], warmup_lr=config['lr_scheduler']['warmup_lr'], 
                warmup=config['lr_scheduler']['warmup'])

    is_distributed = args.world_size > 1
    loris = distribute_model(loris, is_distributed)
    logger.log_info(f"check args.dirtributed: {is_distributed}")

    if args is not None and is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        train_iters = len(train_sampler) // config['batch_size']
        val_iters = len(val_sampler) // config['batch_size']
    else:
        train_sampler = None
        val_sampler = None
        train_iters = len(train_dataset) // config['batch_size']
        val_iters = len(val_dataset) // config['batch_size']

    num_workers = config['num_workers']
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=config['batch_size'], 
                                               shuffle=(train_sampler is None),
                                               num_workers=num_workers, 
                                               pin_memory=True, 
                                               sampler=train_sampler, 
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=config['batch_size'], 
                                             shuffle=False, #(val_sampler is None),
                                             num_workers=num_workers, 
                                             pin_memory=True, 
                                             sampler=val_sampler, 
                                             drop_last=True)

    logger.log_info(f"Finish Initializing DataLoader, Train Dataset Len: {len(train_loader)}, Val Dataset Len: {len(val_loader)}")
    sys.stdout.flush()

    if 'clip_grad_norm' in config:
        clip_grad_norm = instantiate_from_config(config['clip_grad_norm'])
    else:
        clip_grad_norm = None

    #### start training ###
    min_loss = 100.00
    for epoch in range(1, config['max_epochs']+1):
        epoch_start = time.time()
        epoch_loss = 0.0
        step_num = 0
        for batch_idx, input in enumerate(train_loader):
            input['music'], input['motion'], input['video'] = input['music'].cuda(), input['motion'].cuda(), input['video'].cuda()
            step_num += 1
            sys.stdout.flush()
            opt.zero_grad()
            loss = loris(input)
            loss.backward()
            if clip_grad_norm is not None:
                clip_grad_norm(params)
            opt.step()
            scheduler.step(loss)
            epoch_loss += loss
            
        epoch_end = time.time() 
        epoch_avg_loss = epoch_loss / step_num
        eta_sec = (epoch_end-epoch_start)*(config['max_epochs']-epoch-1)
        if epoch % config['log_interval'] == 0:
            logger.log_info('Train Epoch: {}\tLoss: {:.6f}\tEpoch Time: {:.1f}s\tETA: {:2d}h{:2d}min'.format(
                epoch, epoch_avg_loss.item(), epoch_end-epoch_start, int(eta_sec//3600), int((eta_sec%3600)//60)))    
        if epoch % config['save_interval'] == 0 and args.rank == 0:
            save_path = model_save_path + 'loris_epoch' + str(epoch) + ".pt"
            torch.save(loris.state_dict(), save_path)
        if epoch_avg_loss < min_loss:
            save_path1 = model_save_path + "loss_min.pt"
            torch.save(loris.state_dict(), save_path1)
            min_loss = epoch_avg_loss

def main():
    args = parse_args()
    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)

    set_dist_rank(args)

    config_path = os.path.join(args.config_path, args.dataset+'.yaml')
    config = load_yaml_config(config_path)
    # get logger
    logger = Logger(args, config)
    logger.save_config(config)

    train(config, args, logger)
    

if __name__ == '__main__':
    main()


