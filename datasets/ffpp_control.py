import json
import os
import pdb
import random
from tkinter.messagebox import NO
import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import pdb
import glob
from .RandomPatch import RandomPatch
from .transforms import create_data_transforms_alb
import albumentations as ab
import torch.nn.functional as nnf
import clip
try:
    from .data_structure import *
except Exception:
    from data_structure import *
from albumentations.pytorch.transforms import ToTensorV2

class FaceForensicsRelation(Dataset):
    def __init__(self, data_root,
                 num_frames, split, transform=None,base_transform=None,
                 compressions='c23',methods=None,
                  has_mask=False, balance = True,random_patch=None,data_types="",relation_data=False,similarity=False):
        self.data_root = data_root
        self.num_frames = num_frames
        self.split = split

        self.transform = transform
        self.data_types = data_types
        self.base_transform = base_transform
        self.relation_data = relation_data

        self.simlarity = similarity

        if type(random_patch) is int:
            self.random_patch = RandomPatch(grid_size=random_patch)
        else:
            self.random_patch = None


        
        self.compressions = compressions
        self.methods = methods
        self.has_mask = has_mask
        self.mode = "source"

        self.balabce = balance
        self.fake_id_dict = {}


        if self.methods is None:
            self.methods = ['youtube', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

        self.real_items = self._load_items([self.methods[0]])
        self.fake_items = self._load_items(self.methods[1:])

        pos_len = len(self.real_items)
        neg_len = len(self.fake_items)
        print(f'Total number of data: {pos_len+neg_len} | pos: {pos_len}, neg: {neg_len}')

        if self.split == 'train' and self.balabce == True:
            np.random.seed(1234)
            if pos_len > neg_len:
                self.real_items = np.random.choice(self.real_items, neg_len, replace=False).tolist()
            else:
                self.fake_items = np.random.choice(self.fake_items, pos_len, replace=False).tolist()
            image_len = len(self.real_items)
            print(f'After balance total number of data: {image_len*2,} | pos: {image_len}, neg: {image_len}')

        self.items = self.real_items + self.fake_items
        self.items = sorted(self.items, key=lambda x: x['img_path'])
        if relation_data == True:
            self.realtion_item = self._load_relation_items()
        if self.simlarity == True:
            self.get_simlarity()
    def change_mode(self, mode="source"):
        self.mode = mode
    def _load_items(self, methods):
        subdirs = FaceForensicsDataStructure(root_dir=self.data_root,
                                             methods=methods,
                                             compressions=self.compressions,
                                             data_types=self.data_types).get_subdirs()
        splits_path = os.path.join(self.data_root, 'splits')
        video_ids = get_video_ids(self.split, splits_path)
        video_dirs = []
        for dir_path in subdirs:
            video_paths = listdir_with_full_paths(dir_path)
            videos = [x for x in video_paths if get_file_name(x) in video_ids]
            video_dirs.extend(videos)

        items = []
        for video_dir in video_dirs:
            label = 0. if 'original' in video_dir else 1.
            sub_items = self._load_sub_items(video_dir, label)
            items.extend(sub_items)
        
        return items
    def _load_relation_items(self):
        realtion_item = []
        for item in self.items:
            original_path = item['img_path']
            label = item['label']
            video_id = item['video_id']
            frame_id = item['frame_id']
            if label == 0.0:
                
                realtion_item.append({
                    'source_path': item['img_path'],
                    'target_path': item['img_path'],
                    'original_path': item['img_path'],

                    'label': 0.0,
                    'video_id': video_id,
                    'frame_id': frame_id,
                })
            else:
                #如果是假视频的话，需要根据id号找到对应的source和target真视频
                relation_path = original_path.replace("manipulated_sequences","original_sequences")
                relation_path = relation_path.split("/")
                relation_path[-4] = "youtube"
                source_path = relation_path[:]
                target_path = relation_path[:]

                source_path[-2] = video_id.split("_")[0]
                target_path[-2] = video_id.split("_")[1]

                source_path = "/".join(source_path)
                target_path = "/".join(target_path)

                if not os.path.exists(source_path):
                    original_folder = '/'.join(source_path.split('/')[:-1])+'/'
                    face_paths = glob.glob(os.path.join(original_folder, '*.jpg'))
                    source_path = random.choice(face_paths)
                if not os.path.exists(target_path):
                    original_folder = '/'.join(target_path.split('/')[:-1])+'/'
                    face_paths = glob.glob(os.path.join(original_folder, '*.jpg'))
                    target_path = random.choice(face_paths)
                realtion_item.append({
                    'source_path': source_path,
                    'target_path': target_path,
                    'original_path': item['img_path'],
                    'label': 0.0,
                    'video_id': video_id,
                    'frame_id': frame_id,
                })
        return realtion_item
    def _load_sub_items(self, video_dir, label):
        if self.split == 'train' and label == 1:
            num_frames = self.num_frames // 3
        else:
            num_frames = self.num_frames
        video_id = get_file_name(video_dir)
        sorted_images_names = np.array(sorted(os.listdir(video_dir), key=lambda x: int(x.split('.')[0])))
        ind = np.linspace(0, len(sorted_images_names) - 1, num_frames, endpoint=True, dtype=np.int)
        sorted_images_names = sorted_images_names[ind]

        sub_items = []
        for image_name in sorted_images_names:
            frame_id = image_name.split('_')[-1].split('.')[0]
            img_path = os.path.join(video_dir, image_name)
            if 'original' in img_path:
                method = "real" 
            else:
                method = img_path.split("/")[4]
            sub_items.append({
                'img_path': img_path,
                'label': label,
                'video_id': video_id,
                'frame_id': frame_id,
                'method': method
            })
        return sub_items

    
    def get_simlarity(self):
        with open('simlarity.json', 'r') as f:
            self.result_dict = json.load(f)
    def __getitem__(self, index):
        item = self.items[index]
        image_size = 256
        image = cv2.cvtColor(cv2.imread(item['img_path']), cv2.COLOR_BGR2RGB)
        # Resize image to 224x224
        image = cv2.resize(image, (image_size, image_size))
        # Normalize image to [-1, 1]
        
        if self.transform != None:
            image_aug = self.transform(image=image)['image']
        
        if self.random_patch != None and self.split=="train":
            image_aug = ToTensorV2()(image=image_aug)['image']
            image_aug = self.random_patch(image_aug)
            if isinstance(image_aug, torch.Tensor):
                image_aug = image_aug.permute(1, 2, 0).numpy()

        # image = image.astype(np.float32) / 255.0
        image = (image.astype(np.float32) / 127.5) - 1.0
        image = ToTensorV2()(image=image)['image']

        if self.relation_data == True:
            realtion_item = self.realtion_item[index]
            source_image = cv2.cvtColor(cv2.imread(realtion_item['source_path']), cv2.COLOR_BGR2RGB)
            target_image = cv2.cvtColor(cv2.imread(realtion_item['target_path']), cv2.COLOR_BGR2RGB)

            # Resize source_image and target_image to 224x224
            source_image = cv2.resize(source_image, (image_size, image_size))
            target_image = cv2.resize(target_image, (image_size, image_size))
            # Normalize source_image to [0, 1]
            source_image = (source_image.astype(np.float32) / 127.5) - 1.0
            

            target_image = (target_image.astype(np.float32) / 127.5) - 1.0
            

            
            path = item['img_path']
            
            scores = self.result_dict.get(path, 1.0)
            if isinstance(scores, dict):
                source_score = scores['source_score']
                target_score = scores['target_score']
            else:
                source_score = 1.0
                target_score = 1.0
                
            return {
            'source': source_image,
            'target': target_image,
            'txt': '',  # 如果没有对应的文本提示,可以留空或设置为空字符串
            'hint_ori': image,
            'hint': image_aug,
            'label': item['label'],
            'source_score': source_score,
            'target_score': target_score,
            'ori_path': item['img_path']
        }
            
        

    def __len__(self):
        return len(self.items)


def listdir_with_full_paths(dir_path):
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


def get_file_name(file_path):
    return file_path.split('/')[-1]


def read_json(file_path):
    with open(file_path) as inp:
        return json.load(inp)


def get_sets(data):
    return {x[0] for x in data} | {x[1] for x in data} | {'_'.join(x) for x in data} | {'_'.join(x[::-1]) for x in data}


def get_video_ids(spl, splits_path):
    return get_sets(read_json(os.path.join(splits_path, f'{spl}.json')))

