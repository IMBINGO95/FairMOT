# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn
import torch.nn as nn

from M2Det.utils.core import anchors,print_info,init_net

sys.path.append('.')
from Config_ReID import cfg as ReIDCfg
from ReID_model.modeling import build_model

from ReID_model.utils.dataset_loader import read_image
from ReID_model.transforms.build import build_transforms

from ReID_model.modeling import ReID_Model
from utils_BINGO.K_Means import k_means
import time
import shutil


def model_load(ReIDCfg):
	model = build_model(ReIDCfg, 751)
	model.load_param(ReIDCfg.TEST.WEIGHT)
	return model


def img_preprocess(img_path, transform=None):
	img = read_image(img_path)
	if transform is not None:
		img = transform(img)
	return img


def ReID_imgs_load(img_dir):
	val_transforms = build_transforms(ReIDCfg, is_train=False)
	imgs_name = os.listdir(img_dir)
	
	imgs = []
	names = []
	index_to_name = {}
	
	for index, img_name in enumerate(imgs_name):
		if img_name.split('.')[-1] != 'jpg':
			continue
		# need to process damaged img, do something
		img = img_preprocess(os.path.join(img_dir, img_name), val_transforms)
		imgs.append(img.unsqueeze(dim=0))
		names.append(os.path.join(img_dir, img_name))
		index_to_name[index] = img_name.split('.')[0]
	imgs = torch.cat(tuple(imgs), 0)
	
	return imgs,names,index_to_name


def main():
	parser = argparse.ArgumentParser(description="ReID Baseline Inference")
	parser.add_argument(
		"--ReIDCfg",
		default="/datanew/hwb/Re-ID_models/reid-strong-baseline-master/configs/softmax_triplet_with_center.yml",
		help="path to config file", type=str
	)
	
	args = parser.parse_args()
	
	###################################
	# Load in ReID Classification model#
	###################################
	print_info('===> Start to constructing and loading ReID model', ['yellow', 'bold'])
	if args.ReIDCfg != "":
		ReIDCfg.merge_from_file(args.ReIDCfg)
	ReIDCfg.freeze()
	ReID = ReID_Model(ReIDCfg)
	print_info('===> Finished constructing and loading ReID model', ['blue', 'bold'])
	
	root_dir = '/datanew/hwb/data/Football/SoftWare/2/main_imgs'
	dirs = ['/datanew/hwb/data/Football/SoftWare/2/main_imgs']
	
	length = -1
	feats_all = []
	img_names = []
	ass = []
	
	data, names, index_to_name = ReID_imgs_load(root_dir)
	img_names.extend(names)
	
	with torch.no_grad():
		data = data.to('cuda') if torch.cuda.device_count() >= 1 else data
		feats = ReID(data).cpu().numpy().tolist()
	feats_all.extend(feats)
	length += len(feats)
	print('{} end at {}'.format(dir, length))

	t1 = time.time()
	assignments, dataset = k_means(feats_all, 4)
	# for index,cls in enumerate(assignments):
	# 	shutil.copyfile(img_names[index],'/datanew/hwb/data/Football/cluster/test1/{}/{}.jpg'.format(cls,index))
	t2 = time.time()
	ass.append(assignments)
	print('time = {}'.format(t2 - t1))


if __name__ == '__main__':
	# main(
	import cv2
	imgs = cv2.imread('/datanew/hwb/data/Football/SoftWare/2/main_imgs/19.jpg')
	print_info(imgs)
	