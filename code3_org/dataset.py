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

# label_to_index = {
#         'Abuse': 0,
#         'Arrest': 1,
#         'Arson': 2,
#         'Assault': 3,
#         'Burglary': 4,
#         'Explosion': 5,
#         'Fighting': 6,
#         'RoadAccidents': 7,
#         'Robbery': 8,
#         'Shooting': 9,
#         'Shoplifting': 10,
#         'Stealing': 11,
#         'Vandalism': 12,
#         'Normal': 13
#     }

# def load_video(root_dir,annotations,clip_length_in_frames):
#     print('Loading video clips...')
#     video_clips=[]
#     class_index=[]
#     for label, video_path in annotations.itertuples(False):
#             print(label, video_path)
#             video,audio,metadata=torchvision.io.read_video(root_dir + '/' + video_path)
            
#             clips=torch.split(video,clip_length_in_frames,dim=0)
#             for clip in clips:
#                 if clip.shape[0] != clip_length_in_frames:
#                     continue
#                 video_clips.append(clip)
#                 class_index.append(label_to_index[label])

#     return video_clips,class_index

# class ThreadWithReturnValue(Thread):
    
#     def __init__(self, group=None, target=None, name=None,
#                  args=(), kwargs={}, Verbose=None):
#         Thread.__init__(self, group, target, name, args, kwargs)
#         self._return = None

#     def run(self):
#         if self._target is not None:
#             self._return = self._target(*self._args,
#                                                 **self._kwargs)
#     def join(self, *args):
#         Thread.join(self, *args)
#         return self._return
        
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
            num_workers: int=1,
            transform=None):
        
        # annotations = np.array_split(pd.read_csv(annotation_file),num_workers)
        # self.root_dir = root_dir
        # self.transform = transform
        # self.video_clips=[]
        # self.class_index=[]

        # threads = []
        # for i in range(num_workers):
        #     t = ThreadWithReturnValue(target=load_video, args=(root_dir,annotations[i],clip_length_in_frames))
        #     threads.append(t)
        #     t.start()
        #     print('Thread {} started'.format(i))

        # for t in threads:
        #     video_clips,class_index=t.join()
        #     self.video_clips.extend(video_clips)
        #     self.class_index.extend(class_index)

        annotations=pd.read_csv(annotation_file)
        self.video_paths = [root_dir + '/' + video_path for video_path in annotations['Video_path']]
        self.class_index = [self.label_to_index[label] for label in annotations['Video_class']]
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