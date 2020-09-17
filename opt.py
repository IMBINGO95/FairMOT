from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

class OPT_setting(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='SoftWare main ')
        ##########################################
        ###set for data that we want to process!##
        ##########################################
        # set for number classification
        # parser.add_argument('--SvhnCfg', default='config_SVHN/WG_Num_predict.py',type=str)

        self.parser.add_argument('-dr', '--data_root', default='/datanew/hwb/data/Football/SoftWare/143', help='the path to data_root')
        self.parser.add_argument('-dn', '--dir_name', default='143-x', help='the name of the dir(game)')

        self.parser.add_argument('-f','--file_name',default='0.json')
        self.parser.add_argument('-fb','--from_begin', default=True, type=bool, help='If to detection and tracking from begining?')
        self.parser.add_argument('-rp', '--re_pattern',default=r'intermediate\_(\d{1,4})',help='JSON 文件的正则化形式')

        self.parser.add_argument('-fs','--file_save_name',default='with_VOT_')
        self.parser.add_argument('-tl','--track_length',default=25, help='想要追踪的长度')
        self.parser.add_argument('-tkg','--track_gap',default=2,help='每帧进行追踪，但是不需要每帧都保存。间隔 track_gap 帧返回一个追踪结果')
        self.parser.add_argument('-tfg','--transfer_gap',default=3,help='在追踪结果的基础上，每隔 transfer_gap 帧进行一次坐标转换')

        #setting ScreenShot Image parameters
        self.parser.add_argument('-sis','--ImgSize',default=(1088,608),help='The Image Size of ScreenShot ')
        self.parser.add_argument('-sb','--bias',default=200,help='the relative distance between marking point and the lower size of the img')
        self.parser.add_argument('-sr','--radius',default=10,help='The point of marking point')

        # set for person detection
        self.parser.add_argument('--FairMotCfg', default='config/FairMot/defaults.yaml', type=str)

        self.parser.add_argument('--IoUthreshold',default=0.5,type=float,
                            help='if the IoU of the target box is bigger than IoUthreshold, then discard it')

        # set for number classification
        self.parser.add_argument('--SvhnCfg', default='config/SVHN/defaults.yaml',type=str)

        # set for Pose estimation
        # Alphapose poser 的配置文件
        self.parser.add_argument('--PoseEstiCfg',  default='config/alphapose/defaults.yaml', type=str)

        # ReID的配置文件
        self.parser.add_argument(
            "--ReIDCfg",
            default="config/ReID/defaults.yaml",
            help="path to config file", type=str)
        # 根据reID feature，将球员分为几类
        self.parser.add_argument('--num_cls', default=4,
                                 help='how many classes people on the field. A team B team goalkeeper.')

        self.parser.add_argument('-gs', '--gpus', default = '0,1,2,3', help='')
        # 是否将中间结果以可视化的形式保存下来。
        self.parser.add_argument('-v', '--visualization', default=True, help='')


    def init(self):
        opt = self.parser.parse_args()
        opt.file_name = '{}.json'.format(opt.dir_name)
        opt.num_classes = 80
        return opt
