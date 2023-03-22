#
# Video Action Recognition with Pytorch
#
# Paper citation
#
# Action Recognition in Video Sequences using
# Deep Bi-Directional LSTM With CNN Features
# 2017, AMIN ULLAH et al.
# Digital Object Identifier 10.1109/ACCESS.2017.2778011 @ IEEEAccess
#
# See also main.py
#

import requests
import os
import glob
import torch
import pandas as pd

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.video_utils import VideoClips

class UCFCrimeDataset(Dataset):
    # UCF Crime Dataset

    label_to_index = {
        'Abuse': 0,
        'Arrest': 1,
        'Arson': 2,
        'Assault': 3,
        'Burglary': 4,
        'Explosion': 5,
        'Fighting': 6,
        'RoadAccidents': 7,
        'Robbery': 8,
        'Shooting': 9,
        'Shoplifting': 10,
        'Stealing': 11,
        'Vandalism': 12,
        'Normal': 13
    }

    def __init__(
            self,
            root_dir: str,
            annotation_file:str,
            clip_length_in_frames: int,
            frames_between_clips: int=1,
            transform=None):
        self.annotations = pd.read_csv(annotation_file)
        self.root_dir = root_dir
        self.transform = transform
        self.video_clips=[]
        self.class_index=[]

        for label, video_path in self.annotations.itertuples(False):
            print(label, video_path)
            video,audio,metadata=torchvision.io.read_video(root_dir + '/' + video_path)
            
            clips=torch.split(video,clip_length_in_frames,dim=0)
            for clip in clips:
                if clip.shape[0] != clip_length_in_frames:
                    continue
                self.video_clips.append(clip)
                self.class_index.append(self.label_to_index[label])
        
            

    def __len__(self):
        return len(self.video_clips)
    
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        video_clip=self.video_clips[index]
        class_index=self.class_index[index]
        
        if self.transform:
            video_clip=self.transform(video_clip)
        
        return video_clip,class_index

def prepare_dataset(colab):
    if colab:
        base_path = '/content/drive/MyDrive/dataset'
        checkpoints_path = '/content/drive/MyDrive/checkpoints'
        results_path = '/content/drive/MyDrive/results'
    else:
        base_path = '../datasets/ucfcrimedataset'
        checkpoints_path = './checkpoints'
        results_path = './results'    

    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)

    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    
    # if base path as non zero files then return error
    if len(os.listdir(base_path)) == 0:
        print('No dataset found in base path.')
        exit(1)

def to_normalized_float_tensor(video):
    return video.permute(0, 3, 1, 2).to(torch.float) / 255

class ToFloatTensorInZeroOne(object):
    def __call__(self, video):
        return to_normalized_float_tensor(video)