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

from sklearn.metrics import roc_curve, auc
import numpy as np

import os
import shutil
import time
import matplotlib.pyplot as plt
import math
import pickle

from dataset import prepare_dataset
from dataset import ToFloatTensorInZeroOne
from model import LSTM_with_EFFICIENTNET

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

    # timestamp = get_time().replace(':', '')
    timestamp = 'Tue Feb 28 034517 2023'

    location = {
        'video_path': os.path.join(root, '../datasets/hmdb51dataset/video'),
        'annotation_path': os.path.join(root, '../datasets/hmdb51dataset/annotation'),
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
    transform_train = transforms.Compose([
        ToFloatTensorInZeroOne(),
        transforms.Resize([256,342]),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomCrop(224)
    ])
    transform_test = transforms.Compose([
        ToFloatTensorInZeroOne(),
        transforms.Resize([256,342]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.CenterCrop(224)
    ])

    dataset_train_val = torchvision.datasets.HMDB51(
        root=location['video_path'],
        annotation_path=location['annotation_path'],
        frames_per_clip=10,
        step_between_clips=5,
        train=True,
        transform=transform_train
    )
    dataset_test = torchvision.datasets.HMDB51(
        root=location['video_path'],
        annotation_path=location['annotation_path'],
        frames_per_clip=10,
        step_between_clips=5,
        train=False,
        transform=transform_test
    )

    # Train set 70%, validation set 15%, test set 30%
    dataset_len = len(dataset_train_val)
    test_len = len(dataset_test)
    train_len = math.floor(dataset_len * 0.8)
    val_len = dataset_len - train_len
    dataset_train, dataset_val = random_split(dataset_train_val, [train_len, val_len],generator=torch.Generator().manual_seed(42))

    print(train_len,val_len,test_len)
    # num_workers=os.cpu_count()

    # Loading dataset
    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        # num_workers=num_workers
    )
    loader_val = DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        # num_workers=num_workers
    )
    loader_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        # num_workers=num_workers
    )

    train_batches = len(loader_train)
    val_batches = len(loader_val)
    test_batches = len(loader_test)

    ######## Model & Hyperparameters ########
    model = LSTM_with_EFFICIENTNET(num_classes=51,hidden_size=51,num_layers=2,pretrained=True,fine_tune=False).to(device)

    learning_rate = 0.001
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

    ######## Test ########
    print(f"Test | Evaluation start @ {get_time()}", flush=True)

    # Set up lists to store true labels and predicted scores for each class
    true_labels = []
    predicted_scores = []

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for batch_index, (videos, audios, labels) in enumerate(loader_test):
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

            # Store true labels and predicted scores
            predicted_scores.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        test_acc = 100 * correct / total

    print('Test | Accuracy: {:.4f}%'.format(test_acc))
    print(f"Test | Finished evaluation @ {get_time()}", flush=True)

    # ######## Learning Statistics ########
    # if train_epochs == 0:
    #     epoch = done_epochs - 1

    # plt.subplot(2, 1, 1)
    # plt.plot(range(plot_bound + 1, epoch + 2), history['train_loss'], label='Train', color='red', linestyle='dashed')
    # plt.plot(range(plot_bound + 1, epoch + 2), history['val_loss'], label='Validation(Rescaled)', color='blue')

    # plt.title('Loss history')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.plot(range(plot_bound + 1, epoch + 2), history['train_acc'], label='Train', color='red', linestyle='dashed')
    # plt.plot(range(plot_bound + 1, epoch + 2), history['val_acc'], label='Validation', color='blue')

    # plt.title('Accuracy history')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # plt.tight_layout()
    # plt.savefig(os.path.join(location['results_path'], 'result.png'), dpi=1000)

    # Convert true labels and predicted scores to numpy arrays
    true_labels = np.array(true_labels)
    predicted_scores = np.array(predicted_scores)

    # Generate ROC curve and calculate AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(51):
        fpr[i], tpr[i], _ = roc_curve(true_labels == i, predicted_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure()
    lw = 2
    for i in range(51):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve (area = %0.2f) for class %d' % (roc_auc[i], i))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC curve')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(os.path.join(location['results_path'], 'roc.png'), dpi=1000)

    print(f"Code execution done @ {get_time()}", flush=True)

if __name__ == '__main__':
    # Set True when using Google Colab
    colab = False

    # Consider GPU memory limit
    # Paper suggested 512
    # Suggest 128 for GTX 1660 Ti
    batch_size = 24

    # Last checkpoint's training position
    done_epochs = 50

    # Consider Google Colab time limit
    # How much epochs to train now
    train_epochs = 0

    prepare_dataset(colab)
    train_and_eval(colab, batch_size, done_epochs, train_epochs, clear_log=False)
        