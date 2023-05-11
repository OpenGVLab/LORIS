import torch
import subprocess
import numpy as np
import os
import sys
import torch.distributed as dist


def is_primary():
    return get_rank() == 0


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()
    

def distribute_model(model, is_dirtributed):
    """Distribute the model on different machines"""
    if is_dirtributed:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
    else:
        model.cuda()
    return model

def master_only(func):
    """Wrapper that only run on rank=0 node"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)
    return wrapper

def set_dist_rank(args):
    """Set up distributed rank and local rank"""
    ngpus_per_node = torch.cuda.device_count()
    dist_url = 'env://'
    if 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = os.environ['SLURM_NTASKS']
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
        ## manually set is also ok ##
        master_port = str(np.random.randint(29480, 29510))
        os.environ['MASTER_PORT'] = master_port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.world_size = int(ntasks)
        args.rank = int(proc_id)
        args.local_rank = int(proc_id % num_gpus)
        print(f'SLURM MODE: proc_id: {proc_id}, ntasks: {ntasks}, node_list: {node_list}, num_gpus:{num_gpus}, addr:{addr}, master port:{master_port}', flush=True)
    else:
        rank = int(os.getenv('RANK')) if os.getenv('RANK') is not None else 0
        local_rank = int(os.getenv('LOCAL_RANK')) if os.getenv('LOCAL_RANK') is not None else 0
        world_size = int(os.getenv('WORLD_SIZE')) if os.getenv('WORLD_SIZE') is not None else 1
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.cuda.set_device(args.local_rank)
    print('Distributed Init (rank {}): {}'.format(args.rank, dist_url), flush=True)
    torch.cuda.set_device(args.local_rank)
    setup_for_distributed(args.rank==0)
    dist.barrier()

def get_dist_info():
    """Get dist rank and world size from torch.dist"""
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print