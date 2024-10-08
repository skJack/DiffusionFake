
from .ffpp import FaceForensics
from .celeb_df import CelebDF
from .dfd import DeepFakeDetection
from .dfdc import DFDC
from .ffpp_clean import FaceForensicsClean
from .ffpp_control import FaceForensicsRelation
from .deeperforensics import DeeperForensics
from .transforms import create_data_transforms
from .transforms import create_data_transforms_alb
from .transforms import *
from .webface import WebFace
from .wild_deepfake import WildDeepfake
from .celeb_a import CelebA
import torch.utils.data as data
import pdb
from datasets.base_dataset import BaseDataset,MixDataset

'''
    add your dataset here
'''
def create_dataset(args, split):
    
    transform = create_data_transforms(args.transform, split)
    # base_transform = create_data_transforms_alb(args.transform2, split)
    
    kwargs = getattr(args.dataset, args.dataset.name)
    if args.dataset.name == 'ffpp_rela':
        dataset = FaceForensicsRelation(split=split,transform=transform,**kwargs)

    elif args.dataset.name == 'celeb_df':
        dataset = CelebDF(split=split, transform=transform, **kwargs)
    
    

    else:
        raise Exception('Invalid dataset!')

    sampler = None
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)
    shuffle = True if sampler is None and split == 'train' else False
    batch_size = getattr(args, split).batch_size
    if args.debug:
        batch_size = 4
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 sampler=sampler,
                                 num_workers=6,
                                 pin_memory=True,
                                 drop_last = True)
    return dataloader


if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/default.yaml')
    parser.add_argument('--distributed', type=int, default=0)
    args = parser.parse_args()

    local_config = OmegaConf.load(args.config)
    for k, v in local_config.items():
        setattr(args, k, v)

    print('Dataset => ' + args.dataset.name)
    dataloader = create_dataset(args, split='train')
    for i, datas in enumerate(dataloader):
        print(i, datas[0].shape, datas[1].shape)
        break

    dataloader = create_dataset(args, split='val')
    for i, datas in enumerate(dataloader):
        print(i, datas[0].shape, datas[1].shape)
        break

    dataloader = create_dataset(args, split='test')
    for i, datas in enumerate(dataloader):
        print(i, datas[0].shape, datas[1].shape)
        break
    