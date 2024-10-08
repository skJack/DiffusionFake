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
import schedule
import time

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


args = get_parameters()
setup(args)



def inference_dataset(test_dataloader,device,model,ddim_sampler,ddim_steps,num_samples):
    
    y_trues = []
    y_preds = []
    acces = []
    img_paths = []
    for i, datas in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                hint_source = datas['hint'].cuda()
                targets = datas['label'].float().numpy()
                y_trues.extend(targets)
                feature = model.control_model.input_hint_block.forward_features(hint_source) #[bs,320,32,32]
                output = model.control_model.global_pool(feature).flatten(1)
                cls_output = model.control_model.fc(output)
                
                if cls_output.shape[0] == 1:
                    prob = torch.sigmoid(cls_output).cpu().numpy()
                else:
                    prob = torch.sigmoid(cls_output).squeeze().cpu().numpy()
                     
                prediction = (prob >= args.test.threshold).astype(float)
                y_preds.extend(prob)
                acces.extend(targets == prediction)
                
    acc = np.mean(acces)
    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds, pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print(f'#Total# ACC:{acc:.5f}  AUC:{AUC:.5f}  EER:{100*eer:.2f}(Thresh:{thresh:.3f})')

    preds = np.array(y_preds) >= args.test.threshold
    pred_fake_nums = np.sum(preds)
    pred_real_nums = len(preds) - pred_fake_nums
    # print(f"pred dataset:{args.dataset.name},pred id: {args.exam_id},compress:{args.compress}")
    print(f'pred real nums:{pred_real_nums} pred fake nums:{pred_fake_nums}')


def main():
    # Distributed traning
    if args.distributed:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method="env://")
        torch.cuda.set_device(args.local_rank)
        args.world_size = dist.get_world_size()
    

    args.dataset.name = 'celeb_df'
    celeb_test_dataloader = create_dataset(args, split='test')


    # add your other dataset here we use celeb-df as example
    
    # Create model
    device = torch.device("cuda", args.local_rank)
    # Configs

    ddim_steps = 1
    num_samples = 1
    resume_path = args.ckpt_path
    config_path = args.config_path


    model = create_model(config_path)

    model.control_model.define_feature_filter(encoder='tf_efficientnet_b4_ns') #打开特征过滤

    model.cuda()
    model.eval()

    model.load_state_dict(load_state_dict(resume_path, location='cuda'))
    ddim_sampler = DDIMSampler(model)


    print("cdf result")
    inference_dataset(celeb_test_dataloader,device,model,ddim_sampler,ddim_steps,num_samples)




if __name__ == '__main__':

    main()
