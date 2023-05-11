import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from audio_diffusion_pytorch import AudioDiffusionModel, AudioDiffusionConditional, KarrasSchedule, KarrasSampler, AEulerSampler
from d2m.st_gcn.st_gcn_aaai18 import st_gcn_baseline


class MotionEncoder(nn.Module):
    def __init__(self, context_length):
        super().__init__()
        self.context_length = context_length
        in_channels = 3
        d_model = 1024
        layers = 9
        pose_net = st_gcn_baseline(in_channels, d_model, layers=layers, layout='coco', dropout=0.1)
        self.pose_net = pose_net
        self.avgpool1d = nn.AvgPool1d(2, stride=2)

    def forward(self, x):
        '''
        input: bs, context_length, 17, 3
        output: bs, context_length // 2, d_model
        '''
        bs = x.size()[0]        # bs, context_length, 17, 3
        pose = self.pose_net(x.permute(0, 3, 1, 2).unsqueeze(-1))
        out = self.avgpool1d(pose).permute(0, 2, 1)
        return out


class VideoEncoder(nn.Module):
    def __init__(self, context_length, video_dim):
        super().__init__()
        self.context_length = context_length
        self.video_dim = video_dim
        self.lstm = nn.LSTM(video_dim, video_dim//2, 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        '''
        input: bs, context_length, C
        output: bs, context_length, C
        '''
        bs = x.size(0)
        x = x.view(bs, self.context_length, -1)
        h0 = torch.randn(4, bs, self.video_dim//2).to(x.device)
        c0 = torch.randn(4, bs, self.video_dim//2).to(x.device)
        out, hidden = self.lstm(x, (h0, c0))
        self.lstm.flatten_parameters()
        return out


class RhythmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nbins = config['nbins']
        self.win_mean = config['post_avg'] + config['pre_avg']
        self.win_max = config['post_max'] + config['pre_max']
        self.threshold = config['threshold']

    def directogram(self, pose):
        gxy = pose[:, :, :, :2]                     # bs, T, 17, 2, remove confidence
        gxy = gxy.permute(0, 1, 3, 2)               # bs, T, 2, 17
        magnitude = gxy.norm(dim=2)[:, :, None, :]
        phase = torch.atan2(gxy[:, :, 1, :], gxy[:, :, 0, :])
        phase_int = phase * (180 / math.pi) % 180
        phase_int = phase_int[:, :, None, :]
        phase_bin = phase_int.floor().long() % self.nbins
        n, t, c, j = gxy.shape
        out = torch.zeros((n, t, self.nbins, j), dtype=torch.float, device=gxy.device)
        out.scatter_(2, phase_bin, magnitude)
        out = out.sum(3)                                # bs, T, nbins
        return out

    def pick_peak(self, rhy_env):
        bs, n = rhy_env.shape
        rhy_local_mean = rhy_env.unfold(1, self.win_mean, 1).mean(dim=2)
        rhy_local_mean = F.pad(rhy_local_mean, (0, n-rhy_local_mean.size(1)))
        rhy_local_max = torch.max(rhy_env.unfold(1, self.win_max, 1), dim=2)[0]
        rhy_local_max = F.pad(rhy_local_max, (0, n-rhy_local_max.size(1)))
        rhy_global_max = torch.mean(rhy_env, dim=1, keepdim=True).repeat(1, n)
        rhy_peak = ((rhy_local_max - rhy_local_mean) > (0.1 * rhy_global_max)) * (rhy_local_max == rhy_env)
        rhy_peak = rhy_peak.long()
        rhy_peak_mask = F.pad(rhy_peak[:, 1:] - rhy_peak[:, :-1], (0, 1))
        rhy_peak_mask = rhy_peak_mask.bool()
        rhy_peak *= rhy_peak_mask
        return rhy_peak

    def forward(self, pose):
        '''
        input: bs, context_length, 17, 3
        output: rhy_peak: bs, context_length; rhy_env: bs, context_length
        '''
        bs = pose.size(0)                                               # bs, context_length, 17, 3
        motion = pose[:, 1:] - pose[:, :-1]
        # motion = F.pad(motion, (0, 0, 0, 0, 0, 1), mode='constant')   # bs, context_length, 17, 3   
        directo = self.directogram(motion)                              # bs, context_length, K
        sf = directo[:, 1:] - directo[:, :-1]                           # compute spectral flux
        # sf = F.pad(sf, (0, 0, 0, 1), mode="constant")
        sf_abs = (sf + torch.abs(sf)) / 2                               # only consider increase
        rhy_env = torch.sum(sf_abs, dim=2, keepdim=False)               # bs, context_length
        rhy_env = rhy_env / torch.max(rhy_env, dim=1, keepdim=True)[0]  # normalize to 0-1
        rhy_peak = self.pick_peak(rhy_env)
        return rhy_peak, rhy_env.unsqueeze(-1)


class GenreEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_embed = config['num_embed']
        self.embed_dim = config['embed_dim']
        self.genre_embed = nn.Embedding(self.num_embed, self.embed_dim)

    def forward(self, genre):
        '''
        input: bs, num_embed (one-hot vector)
        output: bs, 1, embed_dim 
        '''
        genre_idx = genre.nonzero(as_tuple=True)[1].cuda()
        genre_emb = self.genre_embed(genre_idx).unsqueeze(1)
        return genre_emb


class LORIS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.motion_context_length = config['motion_context_length']
        self.video_context_length = config['video_context_length']
        self.diffusion_length = config['diffusion_length']
        self.diffusion_step = config['diffusion_step']
        self.embedding_scale = config['embedding_scale']
        motion_dim = config['motion_dim']
        video_dim = config['video_dim']
        condition_dim = config['condition_dim']
        self.sample_rate = config['sample_rate']
        self.segment_length = config['segment_length']
        self.vencoder = VideoEncoder(self.video_context_length, video_dim)
        self.genre_dim = None
        # self.mencoder = MotionEncoder(self.motion_context_length)
        self.rencoder = RhythmEncoder(config['rhythm_config'])
        self.use_genre = config['genre_config']['use_genre']
        if self.use_genre:
            self.gencoder = GenreEncoder(config['genre_config'])
            self.genre_dim = config['genre_config']['embed_dim']
        # self.cond_lin = nn.Linear(video_dim+motion_dim, condition_dim)
        self.cond_lin = nn.Linear(video_dim, condition_dim)
        self.autoencoder_type = config['autoencoder_type']
        if self.autoencoder_type == 'diffusion':
            self.diffusion = AudioDiffusionModel(
                in_channels=2, 
                channels=256, 
                patch_blocks=1, 
                patch_factor=32,
                multipliers=[1, 2, 4, 4, 4, 4, 4],
                factors=[4, 4, 4, 2, 2, 2],
                num_blocks=[2, 2, 2, 2, 2, 2],
                attention_heads=16,
            )
        elif self.autoencoder_type == 'cond_diffusion':
            self.cond_diffusion = AudioDiffusionConditional(
                in_channels=2, 
                channels=256, 
                patch_blocks=1, 
                patch_factor=32,
                multipliers=[1, 2, 4, 4, 4, 4, 4],
                factors=[4, 4, 4, 2, 2, 2],
                num_blocks=[2, 2, 2, 2, 2, 2],
                attention_heads=16,
                embedding_max_length=self.video_context_length,
                rhythm_max_length=self.motion_context_length,
                embedding_features=condition_dim,
                genre_features=self.genre_dim,
                embedding_mask_proba=0.1,    # Conditional dropout of batch elements
            )
        else:
            raise NotImplementedError ("Unrecognised Autoencoder Type!")


    def sample(self, input):
        cond_motion_input = input['motion']
        cond_video_input = input['video']
        content = input['music']
        if self.use_genre:
            cond_genre_input = input['genre']
            cond_genre = self.gencoder(cond_genre_input) 
        batch_size = content.size(0)
        cond_video = self.vencoder(cond_video_input)
        cond_rhy_peak, cond_rhy_env = self.rencoder(cond_motion_input)
        cond_rhy_peak = cond_rhy_peak.unsqueeze(-1).float()
        cond_emb = self.cond_lin(cond_video)
        noise = torch.randn(batch_size, 2, 2 ** self.diffusion_length).cuda()
        if self.autoencoder_type == 'diffusion':
            samples = self.diffusion.sample(
                noise=noise,
                num_steps=self.diffusion_step,
            )
        elif self.autoencoder_type == 'cond_diffusion':
            if self.use_genre:
                samples = self.cond_diffusion.sample(
                    noise,
                    num_steps=self.diffusion_step,
                    embedding=cond_emb,
                    embedding_scale=self.embedding_scale, # Classifier-free guidance scale
                    rhythm=cond_rhy_peak,
                    genre=cond_genre,
                )
            else:
                samples = self.cond_diffusion.sample(
                    noise,
                    num_steps=self.diffusion_step,
                    embedding=cond_emb,
                    embedding_scale=self.embedding_scale, # Classifier-free guidance scale
                    rhythm=cond_rhy_peak,
                )                 
        else:
            raise NotImplementedError ("Unrecognised Autoencoder Type!")
        return samples

    def forward(self, input):
        cond_motion_input = input['motion']
        cond_video_input = input['video']
        content = input['music']
        if self.use_genre:
            cond_genre_input = input['genre']
            cond_genre = self.gencoder(cond_genre_input) 

        # cond_motion = self.mencoder(cond_motion_input)
        cond_video = self.vencoder(cond_video_input)
        cond_rhy_peak, cond_rhy_env = self.rencoder(cond_motion_input)
        cond_rhy_peak = cond_rhy_peak.unsqueeze(-1).float()

        # cond_emb = self.cond_lin(torch.cat((cond_video, cond_motion), -1))
        cond_emb = self.cond_lin(cond_video)
        # print(f"check content shape: {content.shape}")
        content = content.expand(content.shape[0], 2, content.shape[2]) # mono to binaural to fit the pre-trained model
        if self.autoencoder_type == 'diffusion':
            loss = self.diffusion(content[:, :, :2 ** self.diffusion_length])
        elif self.autoencoder_type == 'cond_diffusion':
            if self.use_genre:
                loss = self.cond_diffusion(content[:, :, :2 ** self.diffusion_length], embedding=cond_emb, rhythm=cond_rhy_peak, genre=cond_genre)
            else:
                loss = self.cond_diffusion(content[:, :, :2 ** self.diffusion_length], embedding=cond_emb, rhythm=cond_rhy_peak)
        else:
            raise NotImplementedError ("Unrecognised Autoencoder Type!")

        return loss
        