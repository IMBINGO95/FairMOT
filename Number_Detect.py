# -*- coding: utf-8 -*-
# @Time    : 2021/1/29 17:22
# @Author  : BINGO
# @School: 浙江大学
# @Campany: 竺星
# @FileName: Number_Detect.py
import threading, queue, time
from queue import Queue
from threading import Thread, currentThread

import os
from CalibrateTransfer.img_operation import ScreenSHot_batch
from CalibrateTransfer.data_preprocess import write_data_to_json_file, read_data_from_json_file_v2

import numpy as np
import torch.utils.data as data
import torch
import json
import shutil

from ReID_model.modeling import ReID_Model
from utils_BINGO.K_Means import k_means
from ReID_model.utils.dataset_loader import ReID_imgs_load_by_home_and_away

import logging
from utils.log import Log
from utils.timer import Timer
from utils.dir_related_operation import makedir_v1
import cv2
from SVHN.svhn import load_in_Svhn_model
from torchvision import transforms
from PIL import Image
from utils_BINGO.Number_Rectifier import Number_Rectifier
class SVHN_Predict():
    def __init__(self,dir_root, ReIDCfg, Num_Pred_opt, vis, queueSize=1024):

        self.dir_root = dir_root
        self.dir_list = [d for d in os.listdir(self.dir_root) if os.path.isdir(os.path.join(self.dir_root, d))]
        self.dir_list = sorted(self.dir_list, key=lambda x: int(x))
        # logger.info('目标文件夹是{}'.format(self.root_path))
        self.datalen = len(self.dir_list)
        self.Start_Index = 0

        if vis:
            self.vis_path = vis


        # 号码纠正器， 根据四官报告来修改参数
        self.Number_Rectifier = Number_Rectifier

        self.batch_size = 60
        self.Num_Pred_opt = Num_Pred_opt  # 用来设置号码识别模型的参数。
        self.SVHN_predictor = load_in_Svhn_model(self.Num_Pred_opt)

        self.PreProcess_Q = Queue(maxsize=queueSize)  # 在号码识别前，对输入图片进行预处理。
        self.SVHN_Q = Queue(maxsize=queueSize)

        self.transform = transforms.Compose([
            transforms.Resize([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.height_threshold = 21
        self.width_threshold = 12

        # 加载 ReID 模型
        self.ReIDCfg = ReIDCfg
        self.num_cls = 4 # 场上有几种类型的人

        self.logger = Log(__name__, 'SVHN_Predict').getlog()

    def PreProcess_(self):
        self.t_PreProcess = Thread(target=self.PreProcess, args=())
        self.t_PreProcess.daemon = True
        self.t_PreProcess.start()

    def PreProcess(self):
        '''
        对需要号码识别的图片进行预处理。
        '''
        self.logger.debug('The pid of SVHN_Predict.PreProcess() : {}'.format(os.getpid()))
        self.logger.debug('The thread of SVHN_Predict.PreProcess() : {}'.format(currentThread()))
        PreProcess_timer = Timer()

        for dir_index in range(self.Start_Index, self.datalen):
            PreProcess_timer.tic()  # 开始计时
            # self.logger.debug('PreProcess() ======================================== action {}'.format(dir_index))

            this_dir = os.path.join(self.dir_root,self.dir_list[dir_index],'Target')
            imgs_name_list= os.listdir(this_dir)

            if len(imgs_name_list) <= 0:
                self.PreProcess_Q.put((False, (dir_index, [])))
                print('{} is empty'.format(this_dir))
                continue

            imgs_transfered_list = []
            original_imgs = []
            for img_name in imgs_name_list:
                this_img_path = os.path.join(this_dir,img_name)
                this_img = cv2.imread(this_img_path)
                if this_img.size == 0:
                    print('dir_index : {}, img_name : {} is empty'.format(dir_index, img_name) )
                    continue

                height, width, _ = this_img.shape
                if height < self.height_threshold or width < self.width_threshold:
                    # 图片太小了，就不放入骨骼点检测的序列中。
                    continue
                img_transfered = Image.fromarray(this_img)
                img_transfered = self.transform(img_transfered)
                imgs_transfered_list.append(img_transfered)
                original_imgs.append(this_img)
            # 如果都不符合条件的话。
            if len(original_imgs) == 0:
                self.PreProcess_Q.put((False, (dir_index, [])))
            else:
                imgs_transfered_list = torch.stack(imgs_transfered_list, dim=0)
                self.PreProcess_Q.put((True, (dir_index, imgs_transfered_list, original_imgs)))

            # self.logger.info('Calibrate_transfer.sub_img_generate() action {} consums {}s'.format(action_index,sub_img_generate_timer.toc()))
            # self.logger.log(24, 'SVHN_Predict.PreProcess() action {} consums {}s'.format(dir_index, PreProcess_timer.toc()))

    def Predict_(self):
        self.t_Predict = Thread(target=self.Predict, args=())
        self.t_Predict.daemon = True
        self.t_Predict.start()

    def Predict(self):
        '''
        使用 SVHN 对完成预处理的图片进行号码预测
        '''
        Predict_timer = Timer()
        self.logger.debug( 'The pid of SVHN_Predict.Predict() : {}'.format(os.getpid()))
        self.logger.debug( 'The thread of SVHN_Predict.Predict() : {}'.format(currentThread()))

        Number_TrackingID_dict = {}
        for dir_index in range(self.Start_Index, self.datalen):
            Predict_timer.tic() # 开始计时
            Predict_len = 0
            dir_name = self.dir_list[dir_index]
            PreProcess_Flag, PreResults = self.PreProcess_Q.get()
            # self.logger.debug('Predict() ======================================== action {}'.format(action_index))

            if PreProcess_Flag == False:
                # 输入的数据无意义
                preNum = -1

            else:
                # 输入的数据有意义， 读取数据
                _, rectangle_imgs,original_imgs = PreResults
                imgs_length = rectangle_imgs.size(0)
                leftover = 0
                if (imgs_length) % self.batch_size:
                    leftover = 1
                num_batches = imgs_length // self.batch_size + leftover

                if self.vis_path:
                    vis_dir = os.path.join(self.vis_path,'{}'.format(dir_name),'SVHN_Predict')
                    makedir_v1(vis_dir)
                    vis_dir_0 = os.path.join(self.vis_path, '{}'.format(dir_name), 'SVHN_Predict_Minus_one')
                    makedir_v1(vis_dir_0)

                NumsArray = []
                for j in range(num_batches):
                    input_imgs_j = rectangle_imgs[j*self.batch_size:min((j+1)*self.batch_size , imgs_length)]
                    length_logits_j, digits_logits_j = self.SVHN_predictor(input_imgs_j.cuda())

                    '''This max function return two column, the first row is value, and the second row is index '''
                    length_predictions_j = length_logits_j.max(1)[1].cpu().tolist()
                    digits_predictions_j = [digits_logits_j.max(1)[1].cpu().tolist() for digits_logits_j in digits_logits_j]

                    NumsArray_j = []
                    for Num_i in range(len(length_predictions_j)):
                        Number_len = length_predictions_j[Num_i]

                        if Number_len == 1:
                            Num = digits_predictions_j[0][Num_i]
                            NumsArray_j.append(Num)
                        elif Number_len == 2:
                            Num = digits_predictions_j[0][Num_i] * 10 + digits_predictions_j[1][Num_i]
                            NumsArray_j.append(Num)
                        elif Number_len == 0:
                            Num = -1
                            if self.vis_path:
                                cv2.imwrite(os.path.join(vis_dir_0, '{}_P{}.jpg'.format(num_batches*j + Num_i, Num)), original_imgs[Num_i])
                            continue
                        else:
                            continue

                        if self.vis_path:
                            cv2.imwrite(os.path.join(vis_dir, '{}_P{}.jpg'.format(num_batches*j + Num_i, Num)), original_imgs[Num_i])

                    NumsArray.extend(NumsArray_j)
                    Predict_len = len(NumsArray)

                if Predict_len > 1:
                    # NumberArray range from 0 to 99.
                    # We need to count how many times does each number appear!
                    NumsArray = np.histogram(NumsArray, bins=100, range=(0, 100))[0]
                    preNum = np.argmax(NumsArray)
                    # if preNum == 10:
                    #     print('wrong value')
                    preNum_count = NumsArray[preNum]
                    if np.where(NumsArray == preNum_count)[0].size > 1:
                        # if there are more than one number have the maximun counts, then return -1
                        # can sort by number classification scores.
                        preNum = -1
                else:
                    preNum = -1

            # 保存数据
            # self.logger.log(24, 'SVHN_Predict.Predict action {} consums {}s'.format(action_index, Predict_timer.toc()))
            self.logger.log(24,'dir_name {} Predict_len = {} Predict num = {} ============='.format(dir_name, Predict_len, preNum))
            Number_TrackingID_dict[int(dir_name)] = int(preNum)

        with open(os.path.join(self.vis_path,'Number_results.json'),'w') as f :
            json.dump(Number_TrackingID_dict,f)


        self.logger.log(24, '-----------------------------Finished SVHN_Predict.Predict() datalen = {}-----------------------------'.format(self.datalen))






if __name__ == '__main__':

    from opt import OPT_setting

    from Write_Config import readyaml
    from easydict import EasyDict as edict
    opt = OPT_setting().init()

    Num_Pred_opt = edict(readyaml(opt.SvhnCfg))
    ReIDCfg = edict(readyaml(opt.ReIDCfg))
    dir_root = '/datanew/hwb/data/MOT/WestGroundALL/100-s-1/results_pose/ch01'
    vis = '/datanew/hwb/data/MOT/WestGroundALL/100-s-1/Number_vis'
    N_Predictor = SVHN_Predict(dir_root, ReIDCfg, Num_Pred_opt, vis)

    N_Predictor.PreProcess_()
    N_Predictor.Predict_()

    # 等待后处理的线程结束
    N_Predictor.t_PreProcess.join()
    print(24, '----------------Finished N_Predictor.t_PreProcess()----------------')
    N_Predictor.t_Predict.join()
    print(24, '----------------Finished N_Predictor.t_Predict() datalen = {}----------------'.format(
        N_Predictor.datalen))
    # os.kill(os.getpid(),signal.SIGKILL)