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

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import os
import shutil
import time
import matplotlib.pyplot as plt
import math
import pickle

from dataset import prepare_dataset
from dataset import ToFloatTensorInZeroOne
from dataset import HMDB51Dataset
from model_vit import LSTM_with_VIT

def get_time() -> str:
    return time.strftime('%c', time.localtime(time.time()))

def clear_pycache(root: str) -> None:
    if os.path.exists(os.path.join(root, '__pycache__')):
        shutil.rmtree(os.path.join(root, '__pycache__'))

def clear_log_folders(root: str) -> None:
    if os.path.exists(os.path.join(root, 'checkpoints')):
        shutil.rmtree(os.path.join(root, 'checkpoints'))
    if os.path.exists(os.path.join(root, 'history')):
        shutil.rmtree(os.path.join(root, 'history'))
    if os.path.exists(os.path.join(root, 'results')):
        shutil.rmtree(os.path.join(root, 'results'))

# For updating learning rate
def update_learning_rate(optimizer, lr) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_and_eval(colab: bool, batch_size: int, done_epochs: int, train_epochs: int, clear_log: bool = False) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if colab:
        root = '/content/drive/MyDrive'
    else:
        root = './'

    if clear_log:
        clear_log_folders(root)

    ######## Preparing Dataset ########
    print(f"Dataset | Data preparation start @ {get_time()}", flush=True)

    # b0= EfficientNetB0
    # 128= number of lstm cells
    # 2= number of lstm layers
    # 24= number of sizes
    # 15=number of frames per video

    # timestamp = "vit_128_2_24_15_"+get_time().replace(':', '')
    timestamp="vit_128_2_24_15_Thu Apr 27 123401 2023"

    location = {
        'video_path': os.path.join(root, '../datasets/hmdb51dataset/video'),
        'annotation_path': os.path.join(root, '../datasets/hmdb51dataset/hmdb51dataset.csv'),
        'checkpoints_path': os.path.join(root, 'checkpoints', timestamp),
        'history_path': os.path.join(root, 'history', timestamp),
        'results_path': os.path.join(root, 'results', timestamp)
    }
    if not os.path.exists(location['checkpoints_path']):
        os.makedirs(location['checkpoints_path'])

    if not os.path.exists(location['history_path']):
        os.makedirs(location['history_path'])

    if not os.path.exists(location['results_path']):
        os.makedirs(location['results_path'])

    # Preprocessing dataset
    transform = transforms.Compose([
        ToFloatTensorInZeroOne(),
        transforms.Resize([224,224]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_workers=os.cpu_count()
    dataset_val=HMDB51Dataset(location['video_path'],
                                location['annotation_path'],
                                transform=transform,
                                clip_length_in_frames=15,
                                frames_between_clips=15,
                                num_workers=num_workers)

    # Train set 80%, Testing set 20%
    dataset_len = len(dataset_val)
    train_len = math.floor(dataset_len * 0.8)
    test_len = dataset_len - train_len 
    dataset_train, dataset_test = random_split(dataset_val, [train_len,test_len],generator=torch.Generator().manual_seed(69))

    print(train_len,test_len)
    

    # Loading dataset
    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )

    loader_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )

    train_batches = len(loader_train)
    test_batches = len(loader_test)

    ######## Model & Hyperparameters ########
    model = LSTM_with_VIT(num_classes=51,hidden_size=128,num_layers=2,pretrained=True,fine_tune=False).to(device)

    learning_rate = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    plot_bound = 0

    ######## Loading Model ########
    if done_epochs > 0:
        checkpoint = torch.load(os.path.join(location['checkpoints_path'], f"lstm_epoch{done_epochs}.ckpt"), map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        with open(os.path.join(location['history_path'], 'history.pickle'), 'rb') as fr:
            history = pickle.load(fr)
    else:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    
    ######## Train & Validation ########
    print('Train & Validation | Training start @ {}'.format(get_time()), flush=True)

    for epoch in range(done_epochs, done_epochs + train_epochs):
        ######## Train ########
        print('Train | Epoch {:02d} start @ {}'.format(epoch + 1, get_time()), flush=True)

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_index, (videos, labels) in enumerate(loader_train):
            print('Train | Epoch {:02d} | Batch {} / {} start'.format(epoch + 1, batch_index + 1, train_batches), flush=True)
            # videos.shape = torch.Size([batch_size, frames_per_clip, 3, 112, 112])
            # labels.shape = torch.Size([batch_size])

            videos = videos.permute(0, 2, 1, 3, 4)
            # videos.shape = torch.Size([batch_size, 3, frames_per_clip, 112, 112])

            videos = videos.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train loss
            train_loss += loss.item()

            # Train accuracy
            value, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        train_loss/=train_len

        if (epoch + 1) > plot_bound:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

        print('Train | Loss: {:.4f} | Accuracy: {:.4f}%'.format(train_loss, train_acc), flush=True)

        # Save checkpoint
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(location['checkpoints_path'], f"lstm_epoch{epoch + 1}.ckpt"))

        ######## Validation ########
        print('Validation | Epoch {:02d} start @ {}'.format(epoch + 1, get_time()), flush=True)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0

            for batch_index, (videos, labels) in enumerate(loader_test):
                print('Validation | Epoch {:02d} | Batch {} / {} start'.format(epoch + 1, batch_index + 1, test_batches), flush=True)

                # videos.shape = torch.Size([batch_size, frames_per_clip, 3, 112, 112])
                # labels.shape = torch.Size([batch_size])

                videos = videos.permute(0, 2, 1, 3, 4)
                # videos.shape = torch.Size([batch_size, 3, frames_per_clip, 112, 112])

                videos = videos.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(videos)
                loss = criterion(outputs, labels)

                # Validation loss
                val_loss += loss.item()

                # Validation accuracy
                value, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_acc = 100 * correct / total
            val_loss/=test_len

            if (epoch + 1) > plot_bound:
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

            print('Validation | Loss: {:.4f} | Accuracy: {:.4f}%'.format(val_loss, val_acc), flush=True)

        # Decay learning rate
        # if (epoch + 1) % 10 == 0:
        #     learning_rate /= 2
        #     update_learning_rate(optimizer, learning_rate)

        ######## Saving History ########
        with open(os.path.join(location['history_path'], 'history.pickle'),'wb') as fw:
            pickle.dump(history, fw)

    print(f"Train & Validation | Finished training @ {get_time()}", flush=True)

    ######## Test ########
    print(f"Test | Evaluation start @ {get_time()}", flush=True)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for batch_index, (videos, labels) in enumerate(loader_test):
            print('Test | Batch {} / {} start'.format(batch_index + 1, test_batches), flush=True)

            # videos.shape = torch.Size([batch_size, frames_per_clip, 3, 112, 112])
            # labels.shape = torch.Size([batch_size])

            videos = videos.permute(0, 2, 1, 3, 4)
            # videos.shape = torch.Size([batch_size, 3, frames_per_clip, 112, 112])

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)

            value, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total

    print('Test | Accuracy: {:.4f}%'.format(test_acc))
    print(f"Test | Finished evaluation @ {get_time()}", flush=True)

    ######## Learning Statistics ########
    if train_epochs == 0:
        epoch = done_epochs - 1

    plt.subplot(2, 1, 1)
    plt.plot(range(plot_bound + 1, epoch + 2), history['train_loss'], label='Train', color='red', linestyle='dashed')
    plt.plot(range(plot_bound + 1, epoch + 2), history['val_loss'], label='Validation(Rescaled)', color='blue')

    plt.title('Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(range(plot_bound + 1, epoch + 2), history['train_acc'], label='Train', color='red', linestyle='dashed')
    plt.plot(range(plot_bound + 1, epoch + 2), history['val_acc'], label='Validation', color='blue')

    plt.title('Accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(location['results_path'], 'result.png'), dpi=1000)

    print(f"Code execution done @ {get_time()}", flush=True)

if __name__ == '__main__':
    # Set True when using Google Colab
    colab = False

    # Consider GPU memory limit
    # Paper suggested 512
    # Suggest 128 for GTX 1660 Ti
    batch_size = 24

    # Last checkpoint's training position
    done_epochs = 9

    # Consider Google Colab time limit
    # How much epochs to train now
    train_epochs = 11

    prepare_dataset(colab)
    train_and_eval(colab, batch_size, done_epochs, train_epochs, clear_log=False)
        