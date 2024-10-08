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
import torchvision.utils as vutils

import models
from datasets import create_dataset
from utils.logger import Logger
from utils.init import setup
from utils.parameters import get_parameters
from utils.misc import *
import pdb
from share import *
import einops
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn.functional as F
import json
args = get_parameters()
setup(args)

import torch
import matplotlib.pyplot as plt


def compare_vectors(z_source, z_target):
    # 将向量reshape为二维张量 [bs, 4*32*32]
    bs = z_source.shape[0]
    z_source = z_source.view(bs, -1)
    z_target = z_target.view(bs, -1)
    
    # 计算余弦相似度
    similarity = F.cosine_similarity(z_source, z_target, dim=1)
    
    # 将相似度映射到0-1的范围内
    score = (similarity + 1) / 2
    
    return score




def inference_dataset(test_dataloader,device,model,ddim_sampler,ddim_steps,num_samples):

    
    # 创建一个空字典来存储结果
    result_dict = {}

    for i, datas in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            hint_image = datas['hint_ori'].cuda()
            source = datas['source_aug'].cuda()
            target = datas['target_aug'].cuda()
            image_paths = datas['ori_path']  # 获取图像路径

            encoder_posterior_ori = model.encode_first_stage(hint_image)
            encoder_posterior_source = model.encode_first_stage(source)
            encoder_posterior_target = model.encode_first_stage(target)

            z_ori = model.get_first_stage_encoding(encoder_posterior_ori).detach()
            z_source = model.get_first_stage_encoding(encoder_posterior_source).detach()
            z_target = model.get_first_stage_encoding(encoder_posterior_target).detach()

            source_scores = compare_vectors(z_ori, z_source)
            target_scores = compare_vectors(z_ori, z_target)

            # 将得分添加到字典中
            for path, source_score, target_score in zip(image_paths, source_scores, target_scores):
                result_dict[path] = {
                    'source_score': source_score.item(),
                    'target_score': target_score.item()
                }
    with open('simlarity.json', 'w') as f:
        json.dump(result_dict, f)



def main():
    # Distributed traning
    if args.distributed:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method="env://")
        torch.cuda.set_device(args.local_rank)
        args.world_size = dist.get_world_size()
    
    

    # Create dataloader
    ffpp_dataloader = create_dataset(args, split='train')
    
    
    # Create model
    device = torch.device("cuda", args.local_rank)
    # Configs

    resume_path = '/models/control_sd15_ini.ckpt'

    ddim_steps = 10
    num_samples = 2


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('/configs/diffusionfake.yaml')
    
    model.cuda()
    model.train()

    model.load_state_dict(load_state_dict(resume_path, location='cuda'))
    ddim_sampler = DDIMSampler(model)
    inference_dataset(ffpp_dataloader,device,model,ddim_sampler,ddim_steps,num_samples)
    





if __name__ == '__main__':
    main()
