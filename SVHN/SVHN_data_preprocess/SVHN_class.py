import pandas as pd
import json
import torch
import os
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from SVHN_data_preprocess.img_transforms import *
import cv2 as cv
from PIL import Image
from SVHN_data_preprocess.numberRecognize import crop_data_of_img


class SVHNDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = self.json_read(os.path.join(root_dir,json_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame[idx]['name'])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame[idx]['bbox']
        landmarks = np.array(landmarks)
        # landmarks arrange in this format [[left.x, left.y, width,height, label],...]
        landmarks = landmarks.astype('float').reshape(-1, 5)
        bboxes = landmarks[:,0:4]  # (n_objects, 4)
        labels = landmarks[:,4]
        sample = {'image':image,'bboxes':bboxes,'labels':labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def collate_fn(self,batch):
        """
        Since each image may have a different number of numbers, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = list()
        bboxes = list()
        labels = list()

        for b in batch:
            sub_img = torch.tensor(b['image'])
            images.append(sub_img)
            bboxes.append(b['bboxes'])
            labels.append(b['labels'])

        images = torch.stack(images, dim=0)

        '''What kinds of data type that you want to batch together, depends on the data type that you return'''
        return images,bboxes,labels


    def json_read(self,json_file):
        '''transfer json data into numpy array'''
        with open(json_file,'r') as f:
            landmarks = json.load(f)
        return landmarks


class DatasetV1(Dataset):
    def __init__(self, path_to_data_dir,mode,crop = False):
        self.mode = mode
        self.crop = crop
        self.img_dir = os.path.join(path_to_data_dir, mode)
        self.data_file = os.path.join(path_to_data_dir, mode, mode + '.json')
        with open(self.data_file,'r') as f:
            self.data = json.load(f)
        self._length = len(self.data)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        img_name = self.data[index]['name']
        bbox = self.data[index]['bbox']
        image = Image.open(os.path.join(self.img_dir,img_name))

        if self.crop == True:
            region = crop_data_of_img(bbox)
            image = image.crop(tuple(region[0:4]))

        if self.mode == 'test':
            trans_crop = transforms.CenterCrop([54,54])
        else:
            trans_crop = transforms.RandomCrop([54,54])

        transform = transforms.Compose([
            transforms.Resize([64,64]),
            trans_crop,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image = transform(image)
        length = len(bbox)
        digits = [10 for i in range(5)]
        # print('img_name : {}'.format(os.path.join(self.img_dir,img_name)))
        for index in range(length):
            if index >= 5 :
                break
            # print('index : {}'.format(index) )
            digits[index] = bbox[index][-1]
        return image, length, digits

class SJNDataset(Dataset):
    def __init__(self, path_to_data_dir,mode,label_type = 'both',crop = False):
        self.mode = mode
        self.crop = crop
        self.img_dir_pos = os.path.join(path_to_data_dir, mode, mode)
        self.data_file_pos_label = os.path.join(path_to_data_dir, mode, mode + '.json')
        with open(self.data_file_pos_label,'r') as f:
            self.data_pos_label = json.load(f)

        self.img_dir_neg = os.path.join(path_to_data_dir, mode, mode + '_neg_image')
        if os.path.exists(self.img_dir_neg):
            self.data_neg_label = os.listdir(self.img_dir_neg)

        if label_type == 'pos':
            self.data = self.data_pos_label

        elif label_type == 'neg':
            self.data = self.data_neg_label

        else:
            self.data = self.data_pos_label + self.data_neg_label

        self._length = len(self.data)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        sub_data = self.data[index]
        if type(sub_data) == str:
            digits = [10,10]
            length = 0
            image = Image.open(os.path.join(self.img_dir_neg,sub_data))
        else:
            digits = self.data[index][1:3]
            digits = [int(digits[0]),int(digits[1])]
            '''If the first one is 10 then there is no digit in img'''
            if digits[0] == 10:
                length =0
            elif digits[1] == 10:
                length = 1
            else:
                length = 2

            img_name = sub_data[0]
            image = Image.open(os.path.join(self.img_dir_pos,img_name))

        size = image.size

        if self.crop == True:
            '''only get the upper half of the player.'''
            region = (0,0,size[0],int(0.5*size[1]))
            image = image.crop(tuple(region[0:4]))

        if self.mode == 'test':
            trans_crop = transforms.CenterCrop([54,54])
        else:
            trans_crop = transforms.RandomCrop([54,54])

        # IMG = image
        transform = transforms.Compose([
            transforms.Resize([64,64]),
            trans_crop,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image = transform(image)

        return image, length, digits

class SWDataset(Dataset):
    def __init__(self, path_to_data_dir,mode,crop = False):
        self.mode = mode
        self.crop = crop
        self.path_to_data_dir = path_to_data_dir
        self.data_file = os.path.join(path_to_data_dir, 'Software' + '.json')
        with open(self.data_file,'r') as f:
            self.data = json.load(f)
        self._length = len(self.data)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        sub_data = self.data[index]
        x_l = int(sub_data[0])
        y_l = int(sub_data[1])
        x_r = int(sub_data[2])
        y_r = int(sub_data[3])

        '''If the first one is 10 then there is no digit in img'''
        digits = [10,10]
        num = sub_data[5]
        if num> 10 :
            digits[0] = int(num/10)
            digits[1] = num % 10
            length = 2
        else:
            digits[0] = num
            length = 1

        file_name = sub_data[6]
        dir_name= file_name.split('.')[0].split('_')[0]

        img_path = os.path.join(os.path.join(self.path_to_data_dir,dir_name,file_name))
        image = Image.open(img_path)

        '''get the bbox region of the target player.'''
        region1 = (x_l, y_l, x_r, y_r)
        image = image.crop(region1)

        size = image.size

        if self.crop == True:
            '''only get the upper half of the player.'''
            region2 = (0,0,size[0],int(0.5*size[1]))
            image = image.crop(tuple(region2[0:4]))

        if self.mode == 'test':
            trans_crop = transforms.CenterCrop([54,54])
        else:
            trans_crop = transforms.RandomCrop([54,54])

        # IMG = image
        transform = transforms.Compose([
            transforms.Resize([64,64]),
            trans_crop,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image = transform(image)

        sub_data = torch.tensor(sub_data[:-1])

        return image, length, digits, sub_data, file_name

if __name__ == '__main__':
    file_path = '/datanew/hwb/data/Football/SoftWare/From_Software'
    SW = SWDataset(file_path,'train',True)
    for image, length, digits in SW:
        print(length)



