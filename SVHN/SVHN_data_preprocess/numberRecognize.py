#number recognize in muti-view
from SVHNClassifier.model import Model
import cv2
from torch.autograd import Variable
import torch
import time
from PIL import Image
from torchvision import transforms
# from c_transformation.cordinate_t import *
# from c_transformation.function import save_img_Dataset_for_Flag1,fb_cameraCalibrate
import os
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

number_list = [5, 6, 8, 9,10, 14, 15, 17, 19, 20, 21, 28, 33, 39, 60, 66, 69]


def digit_detect(image, model, transformer,max_len):
    '''

    :param image: the image that I want to classify
    :param model: The CNN model that we use
    :param transformer: To transfer the img, usually minus the img with mean values
    :param max_len: the maximun digit length that we want to detect.
    :return:The digit
    '''

    x = preproces(image, transformer) # preproccess the img, and change it to tensor type
    length_logits, digits_logits = model(x)

    length_dict = {}
    for idx, value in enumerate(length_logits[0]):
        if value > 0 and idx <= max_len:
            length_dict[idx] = value

    length_sorted_list = sorted(length_dict.items(), key=lambda x: x[1], reverse=True)

    digits_dict = {}
    for length, l_logits in length_sorted_list:
        this_num = 0
        this_confidence = 0
        for idx_len in range(length + 1):
            digit_confidence = digits_logits[idx_len][0].tolist()
            max_confidence = max(digit_confidence)
            this_confidence += max_confidence
            this_num = this_num * 10 + digit_confidence.index(max_confidence)
        '''confidence include digit mean confidence plus length confidence'''
        digits_dict[length] = (this_num, (l_logits + this_confidence/length).item())
    digits_tuple = sorted(digits_dict.items(),key=lambda  x:x[1],reverse=True)
    return digits_tuple[0][1]

def detect_one_digit(image, model,transformer):

    x = preproces(image, transformer)
    length_logits, digits_logits = model(x)

    digit_confidence = digits_logits[0][0].tolist()
    max_confidence = max(digit_confidence)
    this_num = digit_confidence.index(max_confidence)
    '''confidence include digit mean confidence plus length confidence'''
    return (this_num,max_confidence)

def preproces(image, transformer):
    x = transformer(image)
    return Variable(x.unsqueeze(0)).cuda()

def get_transformer():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transformer =transforms.Compose([
        transforms.Resize([54,54]),
        transforms.ToTensor()
        # normalize
    ])
    return transformer

def find_ID(box,id,p):
    for i in range(len(box)):
        if box[i][p]==id:
            return  i
    return -1

def crop_data_of_img(data):
    '''return the total number of the entir img,and the number region'''
    x_l,y_l,x_r,y_r = [],[],[],[]
    num = 0
    sorted(data,key=lambda x:x[0])
    for sub_data in data:
        num = num*10 + sub_data[-1]
        x_l.append(sub_data[0])
        y_l.append(sub_data[1])
        x_r.append(sub_data[0] + sub_data[2])
        y_r.append(sub_data[1] + sub_data[3])
    return [min(x_l),min(y_l),max(x_r),max(y_r),num]








