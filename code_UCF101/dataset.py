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
import numpy as np
from threading import Thread
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.video_utils import VideoClips

class UCF101Dataset(Dataset):

    def __init__(
            self,
            root_dir: str,
            annotation_file:str,
            clip_length_in_frames: int,
            frames_between_clips: int=1,
            num_workers: int=1,
            transform=None):

        annotations=pd.read_csv(annotation_file)
        self.video_paths = [root_dir + '/' + video_path for video_path in annotations['Video_path']]
        self.class_index = annotations['Video_class'].tolist()
        self.video_clips = VideoClips(self.video_paths, clip_length_in_frames, frames_between_clips,num_workers=num_workers)
        self.transform = transform
            

    def __len__(self):
        return self.video_clips.num_clips()

    
    def __getitem__(self,index):
        video, audio, info, video_idx = self.video_clips.get_clip(index)
        if self.transform:
            video = self.transform(video)
        return video, self.class_index[video_idx]

def prepare_dataset(colab):
    if colab:
        base_path = '/content/drive/MyDrive/dataset'
        checkpoints_path = '/content/drive/MyDrive/checkpoints'
        results_path = '/content/drive/MyDrive/results'
    else:
        base_path = '../datasets/hmdb51dataset'
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