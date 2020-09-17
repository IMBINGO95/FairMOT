
import cv2
import numpy as np
import time
import re
import shutil
import matplotlib.pyplot as plt
import os
import json
import codecs

import random
import time

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

def calculate_biggest_box(img_dir):
	# 同届这个文件夹下图片的长和宽和面积的大小
	img_list = os.listdir(img_dir)
	h = list()
	w = list()
	
	for item in img_list:
		if item[-3:] != 'jpg':
			continue
		img = cv2.imread(os.path.join(img_dir,item))
		
		shape = img.shape
		h.append(shape[0])
		w.append(shape[1])
	h = np.array(h)
	w = np.array(w)
	area = h * w
	plt.figure(figsize=(100,20))
	plt.rcParams.update({'font.size': 60})  # set font size
	
	for index,his in enumerate((w,h,area)):
		plt.subplot(3,1,index+1)
		
		num = 'num:{:->8}\n'.format(len(his))
		max_score = 'max:{:.4f},'.format(np.max(his))
		min_score = 'min:{:.4f}\n'.format(np.min(his))
		mean_score = '(r)mean:{:.4f},'.format(np.mean(his))
		median_score = '(g)median:{:.4f}'.format(np.median(his))
		
		scale = np.histogram(his, bins=500, range=(0,int(np.max(his))+1))
		
		plt.hist(his, bins=500, range=(0,int(np.max(his))+1))
		'''draw mean and median line in the scores histogram'''
		plt.axvline(x=np.mean(his), ymin=np.min(scale[0]), ymax=np.max(scale[0]), linewidth=5, color='r')
		plt.axvline(x=np.median(his), ymin=np.min(scale[0]), ymax=np.max(scale[0]), linewidth=5, color='g')
		
		plt.title(index)
		plt.ylabel('count')
		plt.xlabel(num + max_score + min_score + mean_score + median_score)
		plt.grid(True)
	
	plt.subplots_adjust(hspace=0.5)  # set gap between subplot !
	plt.tight_layout()
	plt.savefig(os.path.join(img_dir, '../1.png'))
	plt.close()
	print('Figure in ', dir, ' saved !')
	
	print('max h = {}, max w = {}'.format(h,w))


def calculate_biggest_box_json(dir):

	json_file = os.path.join(dir,'Annotations.json')
	with open(json_file,'r') as f:
		data=json.load(f)

	h = list()
	w = list()

	for item in data:
		if item['Label'] == 0 :
			continue
		keypoints = item['keypoints']
		# Crop target rectangle
		ul_x = round(float(min(keypoints[3 * 5], keypoints[3 * 11])))
		ul_y = round(float(min(keypoints[3 * 5 + 1], keypoints[3 * 6 + 1])))
		br_x = round(float(max(keypoints[3 * 6], keypoints[3 * 12])))
		br_y = round(float(max(keypoints[3 * 11 + 1], keypoints[3 * 12 + 1])))

		i_width = br_x - ul_x
		i_height = br_y - ul_y
		h.append(i_height)
		w.append(i_width)
	h = np.array(h)
	w = np.array(w)
	area = h * w
	plt.figure(figsize=(100, 20))
	plt.rcParams.update({'font.size': 60})  # set font size

	for index, his in enumerate((w, h, area)):
		plt.subplot(3, 1, index + 1)

		num = 'num:{:->8}\n'.format(len(his))
		max_score = 'max:{:.4f},'.format(np.max(his))
		min_score = 'min:{:.4f}\n'.format(np.min(his))
		mean_score = '(r)mean:{:.4f},'.format(np.mean(his))
		median_score = '(g)median:{:.4f}'.format(np.median(his))

		scale = np.histogram(his, bins=500, range=(0, int(np.max(his)) + 1))

		plt.hist(his, bins=500, range=(0, int(np.max(his)) + 1))
		'''draw mean and median line in the scores histogram'''
		plt.axvline(x=np.mean(his), ymin=np.min(scale[0]), ymax=np.max(scale[0]), linewidth=5, color='r')
		plt.axvline(x=np.median(his), ymin=np.min(scale[0]), ymax=np.max(scale[0]), linewidth=5, color='g')

		plt.title(index)
		plt.ylabel('count')
		plt.xlabel(num + max_score + min_score + mean_score + median_score)
		plt.grid(True)

	plt.subplots_adjust(hspace=0.5)  # set gap between subplot !
	plt.tight_layout()
	plt.savefig(os.path.join(dir, '1.png'))
	plt.close()
	print('Figure in ', dir, ' saved !')

	print('max h = {}, max w = {}'.format(h, w))

def move_file(dir):
	# 将所有的数据分为训练集和测试集。
	Jpeg_dir = os.path.join(dir,'JPEGImages')
	Anno_dir = os.path.join(dir,'Annotations')
	imgs = os.listdir(Jpeg_dir)
	random.shuffle(imgs)
	train_list = imgs[0:8000]
	test_list = imgs[8000:]
	
	Jpeg_train_dir = os.path.join(dir,'train','JPEGImages')
	Anno_train_dir = os.path.join(dir,'train','Annotations')
	
	for item in train_list:
		name = item[:-4]
		shutil.copyfile(os.path.join(Jpeg_dir,item),os.path.join(Jpeg_train_dir,item))
		shutil.copyfile(os.path.join(Anno_dir,name+'.xml'),os.path.join(Anno_train_dir,name+'.xml'))
		
	Jpeg_test_dir = os.path.join(dir,'test','JPEGImages')
	Anno_test_dir = os.path.join(dir,'test','Annotations')
	
	for item in test_list:
		name = item[:-4]
		shutil.copyfile(os.path.join(Jpeg_dir,item),os.path.join(Jpeg_test_dir,item))
		shutil.copyfile(os.path.join(Anno_dir,name+'.xml'),os.path.join(Anno_test_dir,name+'.xml'))

def sort_img(dir,json_file):
	'''
	# 将过于小的图片剔除掉。
	# 将朝向不对的图片剔除掉
	'''

	Jpeg_dir = dir
	imgs_name = os.listdir(Jpeg_dir)

	target_dir = os.path.join(dir,'..','After_sort_Negative_vis')
	os.makedirs(target_dir,exist_ok=True)

	with open(json_file,'r') as f:
		data = json.load(f)

	for item in data:
		image_id = item['image_id']

		img = cv2.imread(os.path.join(Jpeg_dir,image_id))
		H,W = img.shape[0],img.shape[1]

		if H < 130 or W < 60:
			continue

		# keypoints = item['keypoints']
		# l_x = max(keypoints[6*3],keypoints[12*3])
		# r_x = min(keypoints[5*3],keypoints[11*3])
		# if l_x >= r_x:
		# 	continue
		# part_line = {}
		# for n in range(17):
		# 	# v=0 表示这个关键点没有标注（这种情况下x=y=v=0）
		# 	# v=1 表示这个关键点标注了但是不可见(被遮挡了）
		# 	# v=2 表示这个关键点标注了同时也可见
		# 	if item['keypoints'][n * 3 + 2] == 0:
		# 		continue
		# 	elif item['keypoints'][n * 3 + 2] == 1:
		# 		color = GREEN
		# 	# elif item['keypoints'][n * 3 + 2] == 2:
		# 	else:
		# 		color = RED
		#
		# 	cor_x, cor_y = int(item['keypoints'][n * 3 + 0]), int(item['keypoints'][n * 3 + 1])
		# 	part_line[n] = (cor_x, cor_y)
		# 	# cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
		# 	cv2.circle(img, (cor_x, cor_y), 3, color, -1)
		# 	cv2.putText(img, text='{}'.format(n), org=(cor_x, cor_y), fontFace=DEFAULT_FONT, fontScale=0.5, color=BLACK,
		# 				thickness=2)
		# # cv2.putText(img, ''.join(str(human['idx'])), (int(bbox[0]), int((bbox[2] + 26))), DEFAULT_FONT, 1, BLACK, 2)
		# cv2.imwrite(os.path.join(target_dir, image_id),img)
		# print(os.path.join(target_dir, image_id))
		shutil.copyfile(os.path.join(Jpeg_dir, image_id), os.path.join(target_dir, image_id))



def regular_test(root_path):
	# bulid the regular format
	re_videoName = re.compile(r'(c|C)h0\w*.mp4')
	videoNames = {}
	for videoName in os.listdir(root_path):
		if re_videoName.match(videoName):
			videoNames[videoName[3]] = videoName
	print(videoNames)

def T_move_file(path1, path2, target):

	os.makedirs(target,exist_ok=True)
	imgs_name = os.listdir(path1)
	for img in imgs_name:
		if img.split('.')[-1] == 'jpg':
			shutil.copy(os.path.join(path2,img),os.path.join(target,img))

	
	
	
if __name__ == '__main__':

	root_dir = '/datanew/hwb/data/WG_Num/Negative_vis'
	json_file = 'AlphaPose_WG_num.json'

	sort_img(root_dir,os.path.join(root_dir,json_file))
	# sort_img(img_dir)
