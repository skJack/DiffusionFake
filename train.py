#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import wandb
import shutil
import numpy as np
from tqdm import tqdm
from timm.utils import CheckpointSaver
from timm.models import resume_checkpoint
from easydict import EasyDict
from sklearn.metrics import auc, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import copy
from torch.nn.modules.linear import Linear

from wandb.proto.wandb_telemetry_pb2 import Feature

import models
from datasets import create_dataset
from utils.logger import Logger
from utils.init import setup
from utils.parameters import get_parameters
from utils.misc import *
import pdb
from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict



args = get_parameters()
setup(args)
if args.local_rank == 0:
    if args.wandb.name is None:
        args.wandb.name = args.config.split('/')[-1].replace('.yaml', '')
    wandb.init(**args.wandb)
    allow_val_change = False if args.wandb.resume is None else True
    wandb.config.update(args, allow_val_change)
    wandb.save(args.config)
    if len(wandb.run.dir) > 1:
        args.exam_dir = os.path.dirname(wandb.run.dir)
    else:
        args.exam_dir = 'wandb/debug'
        if os.path.exists(args.exam_dir):
            shutil.rmtree(args.exam_dir)
        os.makedirs(args.exam_dir, exist_ok=True)
    shutil.copytree("configs", f'{args.exam_dir}/configs')
    shutil.copytree("datasets", f'{args.exam_dir}/datasets')
    shutil.copytree("models", f'{args.exam_dir}/models')
    shutil.copytree("utils", f'{args.exam_dir}/utils')


    logger = Logger(name='train', log_path=f'{args.exam_dir}/train.log')
    logger.info(args)
    logger.info(args.exam_dir)


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def main():
    # Distributed traning
    if args.distributed:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method="env://")
        torch.cuda.set_device(args.local_rank)
        args.world_size = dist.get_world_size()

    # Create dataloader
    train_dataloader = create_dataset(args, split='train')
    
    # Create model
    device = torch.device("cuda", args.local_rank)
    # Configs
    resume_path = '/models/control_sd15_ini.ckpt'
    logger_freq = 300
    learning_rate = 1e-5

    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('configs/diffusionfake.yaml').cpu()



    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.control_model.define_feature_filter() #open filter

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    logger = ImageLogger(batch_frequency=logger_freq,save_dir=args.exam_dir)
    model_save_dir = os.path.join(args.exam_dir, 'ckpt')
    os.makedirs(model_save_dir, exist_ok=True)

    # 创建 ModelCheckpoint 回调函数
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_save_dir,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        mode='max',
        save_last=True
    )
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger,checkpoint_callback])

    # Train!
    trainer.fit(model, train_dataloader)




if __name__ == '__main__':
    main()
