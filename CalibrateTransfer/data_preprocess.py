import cv2
import json
import codecs
import os
import torch
import time
import shutil
import re
import argparse
# from M2Det.utils.core import print_info

from CalibrateTransfer.class_set import *
from CalibrateTransfer.img_operation import ScreenSHot

# from Config_ReID import cfg as ReIDCfg
# from ReID_model.modeling import ReID_Model
from ReID_model.utils.dataset_loader import ReID_imgs_load_by_home_and_away
#
from utils_BINGO.K_Means import k_means


def make_dir(root_path,index,Secondary_directory='visualization'):
	'''use os module to make a dir
	index:指的是该动作所对应的动作条目
	'''
	visualization_dir = os.path.join(root_path, '{}/{}'.format(Secondary_directory,index))
	if os.path.exists(visualization_dir):
		shutil.rmtree(visualization_dir)
	os.makedirs(visualization_dir)
	return visualization_dir

def for_football_detection(all_img_points,index,save_dir,video_parameter,setting_parameter):
	visualization_dir = make_dir(save_dir, index)
	for count,(action_time,img_point) in enumerate(all_img_points):
		Message = ScreenSHot(img_point, action_time=action_time, video_parameter=video_parameter,
							 setting_parameter=setting_parameter)
		if Message[0] == True:
			cv2.imwrite(os.path.join(visualization_dir,'{}.jpg'.format(count)),Message[1])

def read_subdata(sub_data,Videoparameters):
	'''
	read each item of sub data
	:param sub_data:
	:param Videoparameters:
	:return:
	'''
	channel = sub_data['channel']

	# action_time = sub_data['action_time'] - Videoparameters[channel]['delta_t']
	action_time = sub_data['action_time']
	img_point = [sub_data['image_x'], sub_data['image_y']]
	video_parameter = Videoparameters[channel]
	return channel,action_time,img_point,video_parameter

def regular_videoName(root_path):
	'''
	given an string regular rule, sorted the video names by this regular rule
	:param root_path: the path includes the videos
	:return: the target videoname dictionary.
	'''
	# bulid the regular format
	re_videoName = re.compile(r'(c|C)h0\w*.mp4')
	videoNames = {}
	for videoName in os.listdir(root_path):
		if re_videoName.match(videoName):
			videoNames[videoName[3]] = videoName
	return  videoNames
	
def read_data_from_json_file(root_path,file_name, args):
	'''
	
	:param root_path: root path of the target game
	:param file_name: the file name of target json name
	:param args:
	:return:
	'''
	# read data from json file that operator get.
	with codecs.open(os.path.join(root_path, file_name), 'r', 'utf-8-sig') as f:
		data = json.load(f)
		
	#given an string regular rule, sorted the video names by this regular rule
	videoNames = regular_videoName(root_path)
	
	parameter = data['params']
	action_data = data['data']
	# manully set by operator
	setting_parameter = {'Output_size': args.ImgSize, 'bias': args.bias, 'radius': args.radius}
	Videoparameters = {}
	channel_list = []
	
	for channel in parameter:
		channel_list.append(channel)
		Videoparameter = {}
		Videoparameter['CalibrateParameter'] = dict_To_CalibrateParameter(parameter[channel]['section']['section1'])
		
		# read in video by channel id.
		videoName = channel[-1]
		if channel[-1] in videoNames:
			videoName = videoNames[channel[-1]]
		else:
			raise ValueError('Target video {} does not exits '.format(videoName))
		videoName = os.path.join(root_path,videoName)
		capture =  cv2.VideoCapture(videoName)
		if capture.isOpened():
			Videoparameter['video'] = capture
		else:
			raise ValueError('{} can not oepn'.format(videoName))
		
		Videoparameter['delta_t'] = parameter[channel]['delta_t'] / 1000
		Videoparameters[channel] = Videoparameter

	return Videoparameters,setting_parameter,action_data,channel_list,parameter


def read_data_from_json_file_v2(root_path, file_name, args):
	'''
	v2版本不读取视频
	:param root_path: root path of the target game
	:param file_name: the file name of target json name
	:param args:
	:return:
	'''
	# read data from json file that operator get.
	with codecs.open(os.path.join(root_path, file_name), 'r', 'utf-8-sig') as f:
		data = json.load(f)

	parameter = data['params']
	action_data = data['data']
	# manully set by operator
	setting_parameter = {'Output_size': args.ImgSize, 'bias': args.bias, 'radius': args.radius}
	Videoparameters = {}
	channel_list = []

	for channel in parameter:
		channel_list.append(channel)
		Videoparameter = {}
		Videoparameter['CalibrateParameter'] = dict_To_CalibrateParameter(parameter[channel]['section']['section1'])

		Videoparameter['delta_t'] = parameter[channel]['delta_t'] / 1000
		Videoparameters[channel] = Videoparameter

	return Videoparameters, setting_parameter, action_data, channel_list, parameter

def write_data_to_json_file(root_path,file_name,action_data,parameter, file_save_name='result_'):

	'''write data to json '''
	with codecs.open(os.path.join(root_path, file_name), 'r', 'utf-8-sig') as f:
		data = json.load(f)

	data['data'] = action_data
	with open(os.path.join(root_path,file_save_name+file_name),'w') as f:
		json.dump(data,f)
		
		
def read_stack_data(root, re_format,max_index=0):
	'''to calculate where did the action detection stopped'''
	for file in os.listdir(root):
		groups = re_format.match(file)
		if groups!= None:
			max_index = max(max_index,int(groups[1]))
	return max_index
	

def mk_cluster_dirs(save_dir,num_cls):
	for i in range(num_cls):
		dir = os.path.join(save_dir,str(i))
		if not os.path.exists(dir):
			os.makedirs(os.path.join(dir,'True'))
			os.makedirs(os.path.join(dir,'False'))


def cluster_main_imgs(ReID, ReIDCfg, main_img_dir, action_datas, save_dir, num_cls):
	'''
	:param ReID: ReID model
	:param ReIDCfg: ReID configure
	:param main_img_dir: The dir save the imgs which the programme what to cluster.
	:param action_datas:
	:param save_dir:
	:param num_cls: how many classes that the programme want !
	:return:
	'''
	# make directories to save the clustered imgs.
	mk_cluster_dirs(save_dir, num_cls)

	'''Preprocess the imgs before ReID'''
	if not os.path.exists(main_img_dir):
		raise ValueError("The main_img_dir is not exits")
	imgs_arrays_all, img_names_all = ReID_imgs_load_by_home_and_away(ReIDCfg, main_img_dir, action_datas['data'])

	t1 = time.time()
	cls_res_all = {'Home':0,'Away':2}
	for TeanIndex,TeamType in enumerate(['Home','Away']):
		print('TeamType============================================',TeamType)
		all_feats = []
		imgs_arrays = imgs_arrays_all[TeamType]
		img_names = img_names_all[TeamType]
		cls_res = cls_res_all[TeamType]

		with torch.no_grad():
			if torch.cuda.device_count() >= 1:
				print('torch.cuda.device.count = {}'.format(torch.cuda.device_count()))
			for imgs_array in imgs_arrays:
				imgs_array = imgs_array.to('cuda')
				feats = ReID(imgs_array).cpu().numpy().tolist()
				all_feats.extend(feats)

		length = len(all_feats)
		print('There are {} actions want to be delt with.'.format(dir, length))

		t1 = time.time()
		assignments, dataset = k_means(all_feats, 2)
		# feats = torch.cat(all_feats,dim=0)
		# feats = torch.nn.functional.normalize(feats,dim=1,p=2)
		for index, cls in enumerate(assignments):
			cls += cls_res
			# Is the number of this img detected ?
			if int(action_datas['data'][int(img_names[index])]['num']) == -1 or action_datas['data'][int(img_names[index])]['num'] == None:
				IsNumPredited = False
				shutil.copyfile(os.path.join(main_img_dir, img_names[index] + '.jpg'),
								os.path.join(save_dir,
											 '{}'.format(cls),
											 '{}_.jpg'.format(img_names[index])))
			else:
				IsNumPredited = True
				shutil.copyfile(os.path.join(main_img_dir, img_names[index] + '.jpg'),
								os.path.join(save_dir,
											 '{}'.format(cls),
											 '{}_{}.jpg'.format(img_names[index],
																action_datas['data'][int(img_names[index])]['num'])))

			action_datas['data'][int(img_names[index])]['team'] = str(cls)

	t2 = time.time()
	print('time = {}'.format(t2 - t1))

	return action_datas

def cal_the_accuracy(file):
	with codecs.open(file, 'r', 'utf-8-sig') as f:
		action_datas = json.load(f)
	
	correct = 0
	wrong_10 = 0
	wrong_minus1 = 0
	
	for index, item in enumerate(action_datas['data']):
		if item['num'] == "10":
			wrong_10 += 1
		elif item['num'] == "-1":
			wrong_minus1 += 1
		else:
			correct += 1
	print('wrong number = {}, wrong_minus1 number = {}, correct number = {}'.format(
			wrong_10, wrong_minus1, correct))
		


if __name__ == "__main__":
	
	os.environ["CUDA_VISIBLE_DEVICES"] = '1,5'
	
	main_img_dir = '/datanew/hwb/data/Football/SoftWare/2/main_imgs'
	json_file = '/datanew/hwb/data/Football/SoftWare/2/with_VOT_test_1.json'
	json_file_after_cluster = '/datanew/hwb/data/Football/SoftWare/2/with_VOT_test_1.json'

	
	# with codecs.open(json_file, 'r', 'utf-8-sig') as f:
	# 	action_datas = json.load(f)
	#
	# parser = argparse.ArgumentParser(description='detection and tracking!')
	# # set for ReID classification
	# parser.add_argument(
	# 	"--ReIDCfg",
	# 	default="/datanew/hwb/Re-ID_models/reid-strong-baseline-master/configs/softmax_triplet_with_center.yml",
	# 	help="path to config file", type=str)
	# args = parser.parse_args()
	#
	# ####################################
	# # Load in ReID Classification model#
	# ####################################
	# print_info('===> Start to constructing and loading ReID model', ['yellow', 'bold'])
	# if args.ReIDCfg != "":
	# 	ReIDCfg.merge_from_file(args.ReIDCfg)
	# ReIDCfg.freeze()
	#
	# # imgs_arrays, img_names = ReID_imgs_load(ReIDCfg,main_img_dir)
	#
	#
	# ReID = ReID_Model(ReIDCfg)
	#
	# # ReID = torch.nn.DataParallel(ReID)
	#
	# print_info('===> Finished constructing and loading ReID model', ['blue', 'bold'])
	#
	# data_after = cluster_main_imgs(ReID, ReIDCfg, main_img_dir, action_datas, main_img_dir, 4)
	#
	# with codecs.open(json_file_after_cluster, 'w', 'utf-8-sig') as f:
	# 	json.dump(data_after,f)
