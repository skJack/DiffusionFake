import os
import random
import shutil
import time
import datetime
import warnings
import torch
import numpy as np


def set_seed(SEED):
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True


def setup(args):
    warnings.filterwarnings("ignore")
    os.environ['TORCH_HOME'] = args.torch_home
    set_seed(args.seed)
