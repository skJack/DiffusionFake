import os
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
import pdb
from torchvision import transforms as torch_transforms
import random


class CelebDF(data.Dataset):

    def __init__(self, data_root,split, num_frames,
                 transform=None, base_transform=None,target_transform=None,alb=True,methods = "both",control=False):
        self.split = split
        self.frame_nums = num_frames
        self.transform = transform
        self.target_transform = target_transform
        self.data_root = data_root
        self.alb = alb
        self.methods = methods
        self.control = control
        self.base_transform = base_transform

        self.datas = self._load_items()

        
        
        
        print(f'Total number of data: {len(self.datas)} | pos: {self.real_num}, neg: {self.fake_num}')

    def __getitem__(self, index):
        img_path, target,folder = self.datas[index]

        
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if self.transform != None:
            image_aug = self.transform(image=image)['image']

        if self.control == True:
            image_size = 256
            image = cv2.resize(image, (image_size, image_size))
            # Normalize image to [-1, 1]
            image = image.astype(np.float32) / 255.0
            return{
                'txt': '', 
                'hint': image_aug,
                'label': target,
                'path' : img_path
            }
        if self.base_transform == None:
            if self.transform is not None:
                if self.alb ==  True:
                    image = self.transform(image=image)["image"]
                else:
                    image = self.transform(img=image)
            return image, target,img_path
        else:
            image_norm = self.transform(image=image)['image']
            small_image = self.base_transform(image=image)['image']
            return image_norm,target, small_image,img_path

    def __len__(self):
        return len(self.datas)

    def _load_items(self):
        datas = []
        path_list = []

        Deepfakes_path = f'{self.data_root}/Celeb-synthesis-mtcnn/*'
        original_path = f'{self.data_root}/Celeb-real-mtcnn/*'
        youtube_path = f'{self.data_root}/YouTube-real-mtcnn/*'
        split_path = f'{self.data_root}/List_of_testing_videos.txt'
        data_root = self.data_root
        file = open(split_path)
        with open(split_path,'r') as f:
            self.raw_list = f.read().splitlines()
        test_list = [x.split(" ")[1] for x in self.raw_list]
        
        test_list = [data_root+'/'+x.split("/")[0]+'-mtcnn/'+x.split("/")[1][:-4] for x in test_list]
        if self.methods == "real":
            path_list.append(original_path)
            path_list.append(youtube_path)
        elif self.methods =="fake":
            path_list.append(Deepfakes_path)
        else:
            path_list.append(Deepfakes_path)
            path_list.append(original_path)
            path_list.append(youtube_path)
        self.fake_num = 0
        self.real_num = 0
        if self.split == "train":
            for path in path_list:
                folder_paths_all = glob.glob(path)
                folder_paths = np.setdiff1d(folder_paths_all,test_list)
                label_str = path.split('/')[6]
                
                label = 1.0 if label_str == 'Celeb-synthesis-mtcnn' else 0.0
                for folder in folder_paths:
                    if label == 1.0:
                        self.fake_num = self.fake_num+1
                    else:
                        self.real_num = self.real_num+1
                    face_paths = glob.glob(os.path.join(folder, '*.png'))

                    if len(face_paths) < 5:
                        continue
                    if len(face_paths) > self.frame_nums:
                        face_paths = np.array(sorted(face_paths, key=lambda x: int(x.split('/')[-1].split('.')[0][-4:])))
                        ind = np.linspace(0, len(face_paths) - 1, self.frame_nums, endpoint=True, dtype=np.int)
                        face_paths = face_paths[ind]

                    datas.extend([[face_path, label,folder] for face_path in face_paths])
        else:
            folder_paths = test_list
            for folder in folder_paths:
                label_str = folder.split('/')[6]
                label = 1.0 if label_str == 'Celeb-synthesis-mtcnn' else 0.0
                if self.methods == "real":
                    if label ==1.0:
                        continue
                elif self.methods == "fake":
                    if label ==0.0:
                        continue
                    
                if label == 1.0:
                        self.fake_num = self.fake_num+1
                else:
                        self.real_num = self.real_num+1
                face_paths = glob.glob(os.path.join(folder, '*.png'))
                if len(face_paths) < 5:
                    continue
                if len(face_paths) > self.frame_nums:
                    face_paths = np.array(sorted(face_paths, key=lambda x: int(x.split('/')[-1].split('.')[0][-4:])))
                    ind = np.linspace(0, len(face_paths) - 1, self.frame_nums, endpoint=True, dtype=np.int)
                    face_paths = face_paths[ind]

                datas.extend([[face_path, label,folder] for face_path in face_paths])
            
        return datas

