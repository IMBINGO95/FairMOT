from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import json
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from SVHN_data_preprocess.SVHN_class import SVHNDataset
from SVHN_data_preprocess.img_transforms import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


def show_landmarks(image, landmarks):
    """Show image with landmarks"""

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    l_all = []
    for bbox in landmarks:
        pt1 = [int(bbox[0]),int(bbox[1])]
        pt2 = [int(bbox[0] + bbox[2]),int(bbox[1] + bbox[3])]
        width = int(bbox[2])
        height = int(bbox[3])
        # Create a Rectangle patch
        rect = patches.Rectangle(pt1, width, height, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    """Show image with landmarks"""
    plt.pause(0.001)  # pause a bit so that plots are updated

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, bboxes = \
            sample_batched['image'], sample_batched['bboxes']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        # plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
        #             landmarks_batch[i, :, 1].numpy() + grid_border_size,
        #             s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

def read_data_from_tet(file_name):
    with open(file_name,'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        sub_data = line.split('\t')
        for index,item in enumerate(sub_data):
            if '.jpg' in item:
                continue
            elif '\n' in item :
                sub_data[index] = float(item.split('\n')[0])
            else:
                sub_data[index] = float(item)

        data.append(sub_data)

    return data

if __name__ == '__main__':

    file_name = '/datanew/hwb/data/Football/SoftWare'
    data = read_data_from_tet(file_name)
    with open('/datanew/hwb/data/SJN-210k/test/test.json','w') as f:
        json.dump(data,f)
    print()


    svhn_dataset = SVHNDataset(json_file='test.json',root_dir='/datanew/hwb/data/SVHN/test',
                               transform=transforms.Compose([
                                   Rescale((256,256)),RandomCrop(20)
                               ]))
    #
    # scale = Rescale(256)
    # crop = RandomCrop(20)
    # composed = transforms.Compose([Rescale(256),
    #                                RandomCrop(20)])
    #
    dataloader = DataLoader(svhn_dataset,  batch_size=4, shuffle=True,
                            collate_fn=svhn_dataset.collate_fn, num_workers=4,pin_memory=True)
    #
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched[0].shape,sample_batched[1],sample_batched[2])
    #
    #     # observe 4th batch and stop.
    #     if i_batch == 10:
    #         plt.figure()
    #         show_landmarks_batch(sample_batched)
    #         plt.axis('off')
    #         plt.ioff()
    #         plt.show()
    #         break


