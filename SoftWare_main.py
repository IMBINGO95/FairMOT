from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# import _init_paths
import torch.multiprocessing as mp
import logging
import os
import signal

from FairMot.lib.tracking_utils.utils import mkdir_if_missing
from utils.log import Log
from utils.index_operation import get_index
from utils.timer import Timer,secs_to_clock
from utils.img_operation import Save_sub_imgs

logger = Log(__name__ ,__name__).getlog()
logger.setLevel(logging.INFO)

import cv2
import datetime
import time
from CalibrateTransfer.img_operation import ScreenSHot
import shutil
from threading import currentThread

from FairMot.utils.Loader import FMLoader
from CalibrateTransfer.img_preprocess import Calibrate_transfer
from alphapose.img_operation import Alphapose
from SVHN.core import SVHN_Predict



def get_intermediate_index(opt,If_Restart):
    '''
    根据保存的结果来获取各个模型已经计算到哪一步了。
    '''
    root_path = os.path.join(opt.data_root, '{}'.format(opt.dir_name))
    stack_index = []
    # 初始设置成False， 如果上一部需要重新计算，那么之后的所有步骤也都需要重新计算。
    If_Previous_Restart = False

    for key in If_Restart.keys():
        If_Restart_now = If_Restart[key]
        key_path = os.path.join(root_path,'intermediate_results',key)
        if If_Restart_now == True or If_Previous_Restart == True:
            if os.path.isdir(key_path):
                shutil.rmtree(key_path)
            print('shutil.rmtree({})'.format(key_path))
            If_Previous_Restart = True
            key_index = 0
        else:
            if os.path.isdir(key_path):
                key_index = get_index(key_path)[-1] # 获取已经计算到的最后一个 action_index
            else:
                # 目标文件夹不存在，则从头开始
                key_index = 0
        logger.log(25, 'The {} model has been calculated to step {} '.format(key, key_index))
        stack_index.append(key_index)

    return stack_index

def Short_Track(opt,Tracker_opt,Tracker_output_queue, S_Short_track, S_Coordinate_transfer,in_queueSize, if_vis=False):
    logger.log(21,'The pid of Short_Track() : {}'.format(os.getpid()))

    Tracker = FMLoader(opt,Tracker_opt,Tracker_output_queue,S_Short_track, S_Coordinate_transfer,
                       track_len=1, vis=if_vis,save_results=True, queueSize=in_queueSize,sp=True)
    Tracker.Read_From_Cache()
    Tracker.update_()
    Tracker.PostProcess_()
    # Tracker.get__()

    # 等待后处理的线程结束
    Tracker.t_update.join()
    logger.log(21,'----------------Finished Tracker.t_update()----------------')
    Tracker.t_PostProcess.join()
    logger.log(21,'----------------Finished Tracker.t_PostProcess() datalen = {}-------'.format(Tracker.datalen))
    # os.kill(os.getpid(),signal.SIGKILL)




def Coordinate_transfer(opt,detector_opt,Tracker_output_queue,C_T_output_queue,S_Coordinate_transfer,S_Pose_Estimate,in_queueSize,if_vis=False):

    logger.log(22,'The pid of Coordinate_transfer() : {}'.format(os.getpid()))

    transfer = Calibrate_transfer(opt,detector_opt, Tracker_output_queue,C_T_output_queue,S_Coordinate_transfer,S_Pose_Estimate,
                                  vis=if_vis, save_results=True,queueSize=in_queueSize)
    transfer.Read_From_Cache()

    transfer.update_()
    transfer.detect_()
    transfer.postProcess_()

    # 等待后处理的线程结束
    transfer.t_update.join()
    logger.log(22,'----------------Finished transfer.t_update()----------------')
    transfer.t_detect.join()
    logger.log(22,'----------------Finished transfer.t_detect()----------------')
    transfer.t_postProcess.join()
    logger.log(22,'----------------Finished transfer.t_PostProcess() datalen = {}----------------'.format(transfer.datalen))
    # os.kill(os.getpid(),signal.SIGKILL)

def Pose_Estimate(opt, Pose_opt,ReIDCfg, C_T_output_queue, Pose_output_queue,S_Pose_Estimate, S_Number_Predict,in_queueSize,if_vis=False):
    # 对 sub_imgs 做姿态的检测，基于姿态信息剔除掉部分数据。
    logger.log(23, 'The pid of Pose_Estimate() : {}'.format(os.getpid()))

    Poser = Alphapose(opt, Pose_opt, ReIDCfg, C_T_output_queue, Pose_output_queue, S_Pose_Estimate, S_Number_Predict,
                      vis=if_vis, save_results=True, queueSize=in_queueSize)

    Poser.Read_From_Cache()
    Poser.posing_preprocess_()
    Poser.posing_detect_()
    Poser.posing_postprocess_()

    # 等待后处理的线程结束
    Poser.t_posing_preprocess.join()
    logger.log(23,'----------------Finished Poser.t_posing_preprocess()----------------')
    Poser.t_posing_detect.join()
    logger.log(23,'-------------Finished---Finished Poser.t_posing_detect()----------------')
    Poser.t_posing_postprocess.join()
    logger.log(23,'----------------Finished Poser.t_posing_postprocess() datalen = {}----------------'.format(Poser.datalen))
    # os.kill(os.getpid(),signal.SIGKILL)

def Number_Predict(opt, Num_Pred_opt,ReIDCfg, Pose_output_queue, S_Number_Predict,in_queueSize,if_vis=False):
    # 对 sub_imgs 做姿态的检测，基于姿态信息剔除掉部分数据。
    logger.log(24, 'The pid of Number_Predict() : {}'.format(os.getpid()))

    N_Predictor = SVHN_Predict(opt, ReIDCfg, Num_Pred_opt, Pose_output_queue,S_Number_Predict,
                               vis=if_vis,save_results=True,queueSize=in_queueSize)
    if S_Number_Predict > 0 :
        N_Predictor.Read_From_Cache()

    N_Predictor.PreProcess_()
    N_Predictor.Predict_()

    # 等待后处理的线程结束
    N_Predictor.t_PreProcess.join()
    logger.log(24,'----------------Finished N_Predictor.t_PreProcess()----------------')
    N_Predictor.t_Predict.join()
    logger.log(24,'----------------Finished N_Predictor.t_Predict() datalen = {}----------------'.format(N_Predictor.datalen))
    # os.kill(os.getpid(),signal.SIGKILL)

def main(opt,Tracker_opt,Pose_opt,ReIDCfg,Num_Pred_opt):
    '''主函数'''

    main_timer = Timer()
    main_timer.tic()
    mp.set_start_method('spawn')
    logger.log(20,'The pid of mian() : {}'.format(os.getpid()))
    logger.log(20,'The thread of mian() : {}'.format(currentThread()))
    queueSize = 3

    IF_Restart ={'FMLoader':False,'Calibrate_transfer':False,'Alphapose':False,'SVHN_Predict':True}
    stack_index = get_intermediate_index(opt,IF_Restart)
    [S_Short_track, S_Coordinate_transfer, S_Pose_Estimate,S_Number_Predict] = stack_index

    # [S_Short_track, S_Coordinate_transfer, S_Pose_Estimate,S_Number_Predict] = [114,114,114,0]

    # 根据文件信息来进行追踪。
    Tracker_output_queue = mp.Queue(queueSize)
    # Short_Track 这个函数是用来短时徐追踪人的轨迹并输出ReID的
    Tracker = mp.Process(target=Short_Track,
                         args=(opt,  Tracker_opt, Tracker_output_queue, S_Short_track, S_Coordinate_transfer,queueSize,True))
    Tracker.daemon = True
    Tracker.start()
    # Short_Track(opt,  Tracker_opt, Tracker_output_queue, S_Short_track, S_Coordinate_transfer)

    C_T_output_queue = mp.Queue(queueSize) # C_T : coordinate transfer.
    # Coordinate_transfer(opt, Tracker_opt, Tracker_output_queue, C_T_output_queue,S_Coordinate_transfer,S_Pose_Estimate)
    # 基于追踪数据，将追踪数据转换到其他的视角，并且生成相应的截图何ReID Features
    C_transfer = mp.Process(target=Coordinate_transfer,
                            args=(opt, Tracker_opt, Tracker_output_queue, C_T_output_queue,S_Coordinate_transfer,S_Pose_Estimate,queueSize,True))
    C_transfer.daemon = True
    C_transfer.start()
    #
    #
    Pose_output_queue = mp.Queue(queueSize)
    # Pose_Estimate(opt, Pose_opt, C_T_output_queue, Pose_output_queue, S_Pose_Estimate, S_Number_Predict)
    # 前两步计算得到了一系列sub_imgs, 对这些 sub_imgs 做Pose_Estimate, 并做出相应的更改。
    P_estimate = mp.Process(target = Pose_Estimate,
                            args=(opt, Pose_opt,ReIDCfg, C_T_output_queue, Pose_output_queue, S_Pose_Estimate, S_Number_Predict,queueSize,True))
    P_estimate.daemon = True
    P_estimate.start()
    #
    Number_Predict(opt, Num_Pred_opt,ReIDCfg, Pose_output_queue, S_Number_Predict,queueSize,True)
    # N_predict = mp.Process(target = Number_Predict, args=(opt, Num_Pred_opt, Pose_output_queue, S_Number_Predict,queueSize))
    # N_predict.daemon = True
    # N_predict.start()

    Tracker.join()
    logger.log(25,'----------------Finished tracker process----------------')
    C_transfer.join()
    logger.log(25,'----------------Finished C_transfer process----------------')
    P_estimate.join()
    logger.log(25,'----------------Finished P_estimate process----------------')
    # N_predict.join()
    logger.log(25,'----------------Finished Number_Predict process----------------')

    Total_time = main_timer.toc()
    Total_time = secs_to_clock(Total_time)
    logger.log(31, 'target dir ： {}'.format(opt.dir_name))
    logger.log(31, 'main function consums {}'.format(Total_time))


if __name__ == "__main__":
    # 追踪器的参数
    from opt import OPT_setting
    from Write_Config import readyaml
    from easydict import EasyDict as edict

    opt = OPT_setting().init()

    Tracker_opt = edict(readyaml(opt.FairMotCfg))
    Pose_opt = edict(readyaml(opt.PoseEstiCfg))
    ReIDCfg = edict(readyaml(opt.ReIDCfg))
    Num_Pred_opt = edict(readyaml(opt.SvhnCfg))

    main(opt,Tracker_opt,Pose_opt,ReIDCfg, Num_Pred_opt)
