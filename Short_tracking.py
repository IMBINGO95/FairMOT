from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths

import logging
import os
import os.path as osp
from FairMot.lib.opts import opts
from FairMot.lib.tracking_utils.utils import mkdir_if_missing
from utils.log import Log
logger = Log(__name__,__name__).getlog()
import FairMot.lib.datasets.dataset.jde as datasets
from FairMot.track import Short_track_eval

logger.setLevel(logging.INFO)
import cv2
import datetime
import time
from CalibrateTransfer.img_operation import ScreenSHot
from opt import opt as args
import shutil
from CalibrateTransfer.data_preprocess import write_data_to_json_file,read_data_from_json_file,make_dir,read_subdata,read_stack_data
from CalibrateTransfer.cv_transfer import transform_2d_to_3d,object_To_pixel,updata_img_point
from CalibrateTransfer.img_operation import GenerateRect

from FairMot.lib.tracker.multitracker import JDETracker


def Short_Tracking(opt,game_ID,tracker):
    # define log file
    # 运行程序的日期和时间

    # 比赛场次对应的数据的目录
    root_path = os.path.join(args.data_root, '{}'.format(game_ID))
    logger.info('目标文件夹是{}'.format(root_path))

    file_name = args.file_name

    # read data from json file that software operator made.
    Videoparameters, \
    setting_parameter, \
    action_datas, \
    channel_list, \
    parameter = read_data_from_json_file(root_path, file_name, args)



    for index in range(0, len(action_datas)):
        # if index < 82 :
        # 	continue
        preNum = -1  # 首先假设识别出来的号码为 -1
        print('<===============================================================> action {}'.format(index))
        loop_start = time.time()  # calculate the time .

        # 重置追踪器
        tracker.tracked_stracks = []  # type: list[STrack]
        tracker.lost_stracks = []  # type: list[STrack]
        tracker.removed_stracks = []  # type: list[STrack]

        result_root = make_dir(root_path, index, Secondary_directory='{}_short_tracking'.format(game_ID))

        '''read each item from  subdata of action datas according to the index '''
        channel, action_time, img_point, video_parameter = read_subdata(action_datas[index], Videoparameters)

        video = video_parameter['video']
        video_time = action_time + video_parameter[
            'delta_t']  # action time need to add the delta time to calibrate the time between channels .
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        Message = GenerateRect(img_point, setting_parameter['Output_size'], setting_parameter['bias'], width, height)

        if Message[0] == True:
            # 获取目标区域
            rect = Message[1]
            x_l = int(rect[0])
            y_l = int(rect[1])
            x_r = int(rect[2] + rect[0])
            y_r = int(rect[3] + rect[1])
            rect = [x_l, y_l, x_r, y_r]
            new_point = (int(img_point[0] - x_l), int(img_point[1] - y_l))
            # sub_img = img[y_l:y_r, x_l:x_r]
        else:
            continue

        logger.info('Starting tracking...')
        dataloader = datasets.LoadShortCutVideo(video, video_time, rect, setting_parameter['Output_size'])
        target_frame = dataloader.multiple * dataloader.frame_rate

        result_filename = os.path.join(result_root, '..', '{}.txt'.format(index))
        frame_rate = dataloader.frame_rate
        reference_point = (544,408)
        Short_track_eval(opt, dataloader, 'mot', result_filename,target_frame, reference_point, save_dir=result_root, show_image=False,Input_tracker=tracker, frame_rate=frame_rate)

        # # 保存视频
        # if opt.output_format == 'video':
        #     output_video_path = osp.join(result_root, '..', '{}.mp4'.format(index))
        #     cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(result_root, output_video_path)
        #     os.system(cmd_str)
        #     os.system('y')
        #     shutil.rmtree(result_root)

if __name__ == '__main__':
    import torch.multiprocessing as mp
    opt = opts().init()
    tracker = JDETracker(opt, frame_rate = 25 ) # What is JDE Tracker?

    for game_ID in [1,9]:
        Short_Tracking(opt,game_ID,tracker)
    #
    # num_processes = 4
    # # NOTE: this is required for the ``fork`` method to work
    # # tracker.share_memory()
    # processes = []
    # for game_ID in [1, 9]:
    #     p = mp.Process(target=Short_Tracking, args=(opt,game_ID,tracker))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

