import cv2
import json
import codecs
import os
import torch
import time
import shutil
import re
import argparse

from config.ReID import cfg as ReIDCfg
from ReID_model.modeling import ReID_Model
from ReID_model.utils.dataset_loader import ReID_imgs_load_by_home_and_away

from utils_BINGO.K_Means import k_means
from CalibrateTransfer.data_preprocess import cluster_main_imgs

os.environ["CUDA_VISIBLE_DEVICES"] = '1,5'

dir_index = '143-s'
main_img_dir = '/datanew/hwb/data/Football/SoftWare/{}/intermediate_results/main_imgs'.format(dir_index)
json_file = '/datanew/hwb/data/Football/SoftWare/{}/with_VOT_{}.json'.format(dir_index,dir_index)
json_file_after_cluster = '/datanew/hwb/data/Football/SoftWare/{}/with_VOT_{}_after_team_sort.json'.format(dir_index,dir_index)

with codecs.open(json_file, 'r', 'utf-8-sig') as f:
    action_datas = json.load(f)

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
print('===> Start to constructing and loading ReID model')
if args.ReIDCfg != "":
    ReIDCfg.merge_from_file(args.ReIDCfg)
ReIDCfg.freeze()

ReID = ReID_Model(ReIDCfg)
print('===> Finished constructing and loading ReID model')
data_after = cluster_main_imgs(ReID, ReIDCfg, main_img_dir, action_datas, main_img_dir, 4)

with codecs.open(json_file_after_cluster, 'w', 'utf-8-sig') as f:
    json.dump(data_after, f)
