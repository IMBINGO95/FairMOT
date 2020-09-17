# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
import torch
import torch.nn as nn


def build_model(cfg, num_classes):
	# if cfg.MODEL.NAME == 'resnet50':
	#     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
	model = Baseline(num_classes,
					 cfg.MODEL.LAST_STRIDE,
					 cfg.MODEL.PRETRAIN_PATH,
					 cfg.MODEL.NECK,
					 cfg.TEST.NECK_FEAT,
					 cfg.MODEL.NAME,
					 cfg.MODEL.PRETRAIN_CHOICE)
	return model

def ReID_Model(ReIDCfg,cls_num=751):
	ReID = build_model(ReIDCfg, cls_num)
	ReID.load_param(ReIDCfg.TEST.WEIGHT)
	device = ReIDCfg.MODEL.DEVICE
	if device:
		if torch.cuda.device_count() > 1:
			ReID = nn.DataParallel(ReID)
	ReID.to(device)
	ReID.eval()
	return ReID

