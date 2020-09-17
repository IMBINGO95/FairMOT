
import cv2
import json
import codecs
import os
import torch
import numpy as np
import time

import argparse

from config.ReID import cfg as ReIDCfg
from ReID_model.modeling import ReID_Model

from utils_BINGO.K_Means import k_means
from ReID_model.utils.dataset_loader import transform_imgs,ReID_imgs_loadv1


def imgs_sorted_by_ReID(ReID, ReIDCfg, imgs, distance_threshold=1, feat_norm='yes', version=0, input_features=None, input_gf=None,batch_size=50):
	# preprocess the imgs to ReID model form
	# imgs_array = transform_imgs(ReIDCfg,sub_imgs)
	

	with torch.no_grad():
		# if torch.cuda.device_count() >=1:
		# print('torch.cuda.device.count = {}'.format(torch.cuda.device_count()))
		if type(input_gf) == np.ndarray:
			feats = input_gf
			feats = torch.tensor(feats).cuda()
		else:
			imgs_length = len(imgs)
			leftover = 0
			if (imgs_length) % batch_size :
				leftover = 1
			num_batches = imgs_length // batch_size + leftover
			feats_ = []
			for j in range(num_batches):
				imgs_array_j = transform_imgs(ReIDCfg, imgs[j*batch_size:min((j+1)*batch_size , imgs_length)])
				imgs_array_j = imgs_array_j.to('cuda')
				feats_j = ReID(imgs_array_j)
				feats_.append(feats_j)
			feats = torch.cat(feats_, dim=0)
		if feat_norm == 'yes':
			# print("The test feature is normalized")
			feats = torch.nn.functional.normalize(feats, dim=1, p=2)
		# query : the main sub img is the first one!
		if version == 0:
			qf = feats[:1]
		elif version == 1:
			qf = torch.mean(feats, dim=0)
			qf = torch.unsqueeze(qf,dim=0)
		elif version == 2:
			qf = input_features
		# q_pids : np.asarray(self.pids[:self.num_query])
		# q_camids = np.asarray(self.camids[:self.num_query])
		# gallery : means the other sub imgs

		gf = feats[:]
		# g_pids = np.asarray(self.pids[self.num_query:])
		# g_camids = np.asarray(self.camids[self.num_query:])
		m, n = qf.shape[0], gf.shape[0]
		# print('m = {}, n = {}'.format(m, n))
		# calculate the distance between the query and gallery.
		# distance = x^2 + y^2 -2*x*y
		distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
		          torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
		distmat.addmm_(1, -2, qf, gf.t())
		distmat = distmat.cpu().numpy()
		# imgs_index is a tuple
		imgs_index = np.where(distmat < distance_threshold)

	return imgs_index[1].tolist(), distmat, qf

	
if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '1,5'
	import shutil

	parser = argparse.ArgumentParser(description='detection and tracking!')
	# set for ReID classification
	parser.add_argument(
		"--ReIDCfg",
		default="/datanew/hwb/Re-ID_models/reid-strong-baseline-master/configs/softmax_triplet_with_center.yml",
		help="path to config file", type=str)
	args = parser.parse_args()
	
	####################################
	# Load in ReID Classification model#
	####################################
	print('===> Start to constructing and loading ReID model', ['yellow', 'bold'])
	if args.ReIDCfg != "":
		ReIDCfg.merge_from_file(args.ReIDCfg)
	ReIDCfg.freeze()
	
	# imgs_arrays, img_names = ReID_imgs_load(ReIDCfg,main_img_dir)
	
	ReID = ReID_Model(ReIDCfg)
	ReID.cuda()
	# ReID = torch.nn.DataParallel(ReID)
	print('===> Finished constructing and loading ReID model', ['blue', 'bold'])

	# json_file = '/datanew/hwb/data/Football/SoftWare/9/with_VOT_test_1.json'
	# with codecs.open(json_file, 'r', 'utf-8-sig') as f:
	# 	action_datas = json.load(f)
	# 需要分类的
	# json_file_after_cluster = '/datanew/hwb/data/Football/SoftWare/0/with_VOT_test_0.json'
	game_id = 0
	input_gf = False
	wrong_index = [5 for i in range(100)]


	vis_dir_main = '/datanew/hwb/data/Football/SoftWare/{}/intermediate_results/vis_ReID/'.format(game_id)
	if os.path.isdir(vis_dir_main):
		shutil.rmtree(vis_dir_main)

	for dir_index in wrong_index:
		print(dir_index)
		main_img_dir = '/datanew/hwb/data/Football/SoftWare/{}/intermediate_results/Calibrate_transfer/{}/tracking'.format(game_id,dir_index)
		feature_path = '/datanew/hwb/data/Football/SoftWare/{}/intermediate_results/Calibrate_transfer/{}/tracking/{}_ReID_features_tracking.npy'.format(game_id,dir_index,dir_index)
		feature = np.load(feature_path)
		# feature = torch.tensor(feature).cuda()
		main_names = [name for name in os.listdir(main_img_dir) if name.split('.')[-1]=='jpg']
		main_names = sorted(main_names, key=lambda x:int(x.split('.')[0]))
		imgs = [cv2.imread(os.path.join(main_img_dir,img_name)) for img_name in main_names]

		main_img_dir_detection = '/datanew/hwb/data/Football/SoftWare/{}/intermediate_results/Calibrate_transfer/{}/detection'.format(game_id,dir_index)
		feature_path_detection = '/datanew/hwb/data/Football/SoftWare/{}/intermediate_results/Calibrate_transfer/{}/detection/{}_ReID_features_detection.npy'.format(game_id, dir_index,dir_index)
		feature_detection = np.load(feature_path_detection)
		# feature_detection = torch.tensor(feature_detection).cuda()

		main_names_detection = [name for name in os.listdir(main_img_dir_detection) if name.split('.')[-1] == 'jpg']
		main_names_detection = sorted(main_names_detection, key=lambda x: int(x.split('.')[0]))
		imgs_detection = [cv2.imread(os.path.join(main_img_dir_detection, img_name)) for img_name in main_names_detection]


		vis_dir = '/datanew/hwb/data/Football/SoftWare/{}/intermediate_results/vis_ReID/{}'.format(game_id,dir_index)
		if input_gf == True:
			imgs_index, distmat, output_feature = imgs_sorted_by_ReID(ReID, ReIDCfg, imgs, distance_threshold=1, feat_norm='yes', version=0, input_gf=feature)
		else:
			imgs_index, distmat, output_feature = imgs_sorted_by_ReID(ReID, ReIDCfg, imgs, distance_threshold=1, feat_norm='yes', version=0, input_gf=None)
		Positive_dir = os.path.join(vis_dir,'')
		os.makedirs(Positive_dir,exist_ok=True)
		Negative_dir = os.path.join(vis_dir,'Negative')
		os.makedirs(Negative_dir,exist_ok=True)

		for P_index, _ in enumerate(imgs):
			distance = distmat[0,P_index]
			if P_index in imgs_index:
				cv2.imwrite(os.path.join(Positive_dir,'{}_{:3f}.jpg'.format(P_index,distance)),imgs[P_index])
			else:
				cv2.imwrite(os.path.join(Negative_dir,'{}_{:3f}.jpg'.format(P_index,distance)),imgs[P_index])


		if input_gf == True:
			imgs_index_detection, distmat_detection, _ = imgs_sorted_by_ReID(ReID, ReIDCfg, imgs_detection, distance_threshold=1, feat_norm='yes',
														  version=2,input_features=output_feature,input_gf=feature_detection)
		else:
			imgs_index_detection, distmat_detection, _ = imgs_sorted_by_ReID(ReID, ReIDCfg, imgs_detection,
																			 distance_threshold=1, feat_norm='yes',
																			 version=2, input_features=output_feature,
																			 input_gf=None)

		Positive_dir_detection = os.path.join(vis_dir, 'detection')
		os.makedirs(Positive_dir_detection, exist_ok=True)
		Negative_dir_detection = os.path.join(vis_dir,'detection', 'Negative')
		os.makedirs(Negative_dir_detection, exist_ok=True)

		for P_index_detection, _ in enumerate(imgs_detection):
			distance = distmat_detection[0,P_index_detection]
			if P_index_detection in imgs_index_detection:
				cv2.imwrite(os.path.join(Positive_dir_detection, '{}_{:3f}.jpg'.format(P_index_detection,distance)), imgs_detection[P_index_detection])
			else:
				cv2.imwrite(os.path.join(Negative_dir_detection, '{}_{:3f}.jpg'.format(P_index_detection,distance)), imgs_detection[P_index_detection])

