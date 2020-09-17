import os
import sys
from threading import Thread, currentThread
from queue import Queue

import cv2
import scipy.misc
import numpy as np
import json
import gc
import sys
import copy

from CalibrateTransfer.img_operation import ScreenSHot
from CalibrateTransfer.data_preprocess import write_data_to_json_file,read_data_from_json_file,make_dir,read_subdata,read_stack_data
from CalibrateTransfer.cv_transfer import transform_2d_to_3d,object_To_pixel,updata_img_point
from CalibrateTransfer.img_operation import GenerateRect

import torch
import torch.multiprocessing as mp

from FairMot.lib.tracker.multitracker import JDETracker, create_JDETracker_model

from FairMot.track import Short_track

from utils.sort_by_point import sort_by_point
from utils.log import Log
from utils.timer import Timer
from utils.dir_related_operation import makedir_v1
from utils.timer import show_memory_info


class LoadShortCutVideo:  # for short tracking
    def __init__(self,video, video_time, rect, Output_size, img_size=(1088, 608), multiple = 2):

        self.cap = video
        self.cap.set(cv2.CAP_PROP_POS_MSEC,round(1000*video_time))  # 将视频设置到动作发生的时间
        self.current_frame_index = self.cap.get(cv2.CAP_PROP_POS_FRAMES) # 动作发生的时间对应的帧
        self.multiple = multiple # 获取 2 * multiple倍 的视频帧率长度的图片
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS))) # 计算帧率
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index - multiple*self.frame_rate) # 将视频向前调整 multiple 秒

        self.width, self.height = img_size[0] , img_size[1] # 网络输入的Feature Map的大小
        [self.vw, self.vh] = Output_size # 输入图片的大小
        [self.w, self.h] = Output_size # 可视化的图片的大小

        self.rect = rect # 对应的目标区域 [x_l,y_l,x_r,y_r]
        self.count = 0
        self.vn = 2 *multiple * self.frame_rate + 1
        # print('Lenth of the video: {:d} frames'.format(self.vn))

    def __iter__(self):
        self.count = -1
        return self

    # def __del__(self):
    #     print("调用__del__() LoadShortCutVideo，释放其空间")

    def __next__(self):
        # Read image
        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
        # 裁剪图片
        img0 = img0[self.rect[1]:self.rect[3],self.rect[0]:self.rect[2]]
        # Normalize RGB
        img = img0[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        self.count += 1

        if self.count == len(self):# 结束迭代
            raise StopIteration

        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of files

class FMLoader:
    '''
    载入FairMot的短时序追踪结果
    '''
    def __init__(self, opt,tracker_opt,Tracker_output_queue, S_Short_track, S_Coordinate_transfer, track_len=2, vis=False,
                 save_results=False, queueSize=1000, sp=False):

        self.logger = Log(__name__, 'FMLoader').getlog()

        self.opt = opt
        self.track_len = track_len # 相对于动作点，前后追踪 track_len 秒

        self.dir_name = opt.dir_name
        self.root_path = os.path.join(opt.data_root, '{}'.format(opt.dir_name))

        # 从缓存中读取已经计算出来的结果。
        self.S_Short_track = S_Short_track #代表的是一个index， 在这个数之前追踪结果的都已经计算并保存了。
        self.S_Coordinate_transfer = S_Coordinate_transfer #代表的是一个index， 在这个数之前转换结果的都已经计算并保存了。

        # logger.info('目标文件夹是{}'.format(self.root_path))
        self.file_name = opt.file_name
        # 本来就是要载入两次视频，分开读亦可以
        self.Videoparameters, \
        self.setting_parameter, \
        self.action_datas, \
        self.channel_list, \
        self.parameter = read_data_from_json_file(self.root_path, self.file_name, self.opt)

        self.datalen = len(self.action_datas)
        self.logger.log(21,'___________________一共有 {} 条数据_______________'.format(self.datalen))

        # 是否要将图画出来，可视化给别人看
        self.vis = vis
        if self.vis == True:
            self.vis_path = os.path.join(self.root_path, 'vis')
            os.makedirs(self.vis_path, exist_ok=True)

        self.save_results = save_results
        if self.save_results == True:
            self.intermediate_results_dir = os.path.join(self.root_path, 'intermediate_results','FMLoader')
            os.makedirs(self.intermediate_results_dir, exist_ok=True)

        self.tracker_opt = tracker_opt # 用来设置追踪器参数的。
        self.IoUthreshold = 0.5 #

        self.logger.info('Creating model...')
        self.tracker_model = create_JDETracker_model(self.tracker_opt)
        # self.tracker = JDETracker(self.tracker_opt ) # What is JDE Tracker?

        # initialize the queue used to store frames read from
        # the video file

        self.PostProcess_Q = Queue(maxsize=queueSize)
        self.Output_Q =Tracker_output_queue

    def Read_From_Cache(self):
        '''
        从文件把之前计算过的结果提取出来
        '''
        from utils.index_operation import get_index

        self.logger.debug('The pid of FMLoader.Read_From_Cache() : {}'.format(os.getpid()))
        self.logger.debug('The thread of FMLoader.Read_From_Cache() : {}'.format(currentThread()))

        cache_index = get_index(self.intermediate_results_dir)
        # 只需读取有用的部分即可。
        action_index = self.S_Coordinate_transfer
        for action_index in range(self.S_Coordinate_transfer,self.S_Short_track):

            if action_index not in cache_index:
                # cache 中没有保存说明 此动作本身是False
                self.Output_Q.put((False,action_index,[]))
            else:
                # 从文件夹中读取出该动作对应的计算结果。
                _, [frames_time, sub_imgs, ReID_feature_list, bottom_center_point_list] = self.load_intermediate_resutls(action_index)
                self.Output_Q.put((True, action_index, [frames_time, sub_imgs, ReID_feature_list, bottom_center_point_list]))

        self.logger.log(21, 'length of self.Output_Q = {}'.format(self.Output_Q.qsize()))
        self.logger.log(21, ' FMLoader loads action {} from Cache file '.format(action_index))
        # show_memory_info('===========Read_From_Cache==============')

    def update_(self):

        self.t_update = Thread(target=self.update, args=())
        self.t_update.daemon = True
        self.t_update.start()

        return self

    def update(self):
        '''
        '''
        self.logger.debug( 'The pid of FMLoader.update_() : {}'.format(os.getpid()))
        self.logger.debug( 'The thread of FMLoader.update() : {}'.format(currentThread()))
        self.logger.log(21, 'self.datalen  : {}'.format(self.datalen))
        self.update_timer = Timer()

        # keep looping the whole dataset
        for index in range(self.S_Short_track,self.datalen):

            self.update_timer.tic()
            self.logger.debug('update  <===========> action {} '.format(index))
            # show_memory_info('action _ {}, {}'.format(index, ' S_Short_track begin'))

            # result_root = make_dir(self.root_path, index, Secondary_directory='{}_short_tracking'.format(self.dir_name))
            '''read each item from  subdata of action datas according to the index '''
            channel, action_time, img_point, video_parameter = read_subdata(self.action_datas[index], self.Videoparameters)

            video = video_parameter['video']
            # action time need to add the delta time to calibrate the time between channels .
            video_time = action_time + video_parameter['delta_t']
            width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            Message = GenerateRect(img_point, self.setting_parameter['Output_size'], self.setting_parameter['bias'], width,
                                   height)

            if Message[0] == True:
                # 获取目标区域
                rect = Message[1]
                x_l = int(rect[0])
                y_l = int(rect[1])
                x_r = int(rect[2] + rect[0])
                y_r = int(rect[3] + rect[1])
                rect = [x_l, y_l, x_r, y_r]
                # 目标点坐标相对于从原图中的位置，更新到相对于截图中的位置
                reference_point = (int(img_point[0] - x_l), int(img_point[1] - y_l))
                Left_Up_points = (rect[0],rect[1])  # 截图的右上角相对于原图的位置
                # sub_img = img[y_l:y_r, x_l:x_r]
            else:
                # 如果没有截图则,则无需放入Queue中。
                self.PostProcess_Q.put((False, None, None, None, None ,None,None))
                continue

            # 没有必要逐帧检测。
            # print('self.setting_parameter[\'Output_size\'], multiple=self.track_len', self.setting_parameter['Output_size'], self.track_len)
            self.logger.debug('Starting Building LoadShortCutVideo...')

            dataloader = LoadShortCutVideo(video, video_time, rect, self.setting_parameter['Output_size'], multiple=self.track_len)
            # show_memory_info('action _ {}, {}'.format(index, 'LoadShortCutVideo release ==========='))

            target_frame = dataloader.multiple * dataloader.frame_rate
            frame_rate = dataloader.frame_rate
            start_time = action_time - self.track_len # 调整到需要追踪的小视频，相对于开球的时间。

            # 进行短时徐追踪
            self.logger.debug('Starting tracking...')
            tracker = JDETracker(self.tracker_opt, self.tracker_model) # 创建一个追踪器
            results = Short_track(tracker, dataloader, self.tracker_opt)

            # 删除tracker并立即会手内存
            # del tracker
            # gc.collect()

            # 传入Queue中。
            self.PostProcess_Q.put((True,results, target_frame, start_time, frame_rate, Left_Up_points, reference_point))
            # del results
            # show_memory_info('action _ {}, {}'.format(index, 'self.PostProcess_Q.put'))

            self.logger.log(21, 'FMLoader.update() action {} consums {}s'.format(index,self.update_timer.toc()))

    def PostProcess_(self):
        self.t_PostProcess = Thread(target=self.PostProcess, args=())
        self.t_PostProcess.daemon = True
        self.t_PostProcess.start()

    def PostProcess(self):
        '''
        数据结果后处理
        '''
        self.PostProcess_timer = Timer()
        self.logger.debug('The pid of FMLoader.PostProcess : {}'.format(os.getpid()))
        self.logger.debug('The thread of FMLoader.PostProcess : {}'.format(currentThread()))

        for action_index in range(self.S_Short_track, self.datalen):
            self.PostProcess_timer.tic()

            # show_memory_info('action _ {}, {}'.format(action_index, 'Before get '))
            Flag , results, target_frame, start_time, frame_rate, Left_Up_points, reference_point = self.PostProcess_Q.get()
            self.logger.debug('PostProcess <===========> action {} '.format(action_index))

            # 截图都没有。
            if Flag == False:
                self.Output_Q.put((False,action_index,[]))
                continue


            # 把每个sub_box提取出来。
            sub_imgs = []
            ReID_feature_list = []
            frames_time = []
            bottom_center_point_list = []

            for bias in [0, -1, 1, -2, 2]:  # 总能检测到的?
                input_result = results[target_frame + bias]
                if len(input_result[1]) == 0:  # 有可能目标帧没有检测到，一个目标都没有。
                    target_id = None
                    continue
                new_reference_point, target_id = sort_by_point(results[target_frame + bias], reference_point,IoUthreshold=self.IoUthreshold)

                if target_id != None:
                    # 检测到了的话，就跳出循环
                    # 将目标帧的图片放在了sub_imgs 和 ReID_feature_list 队列的首个
                    bboxes = input_result[1]
                    ids = input_result[2]
                    target_id_index = ids.index(target_id)
                    box = bboxes[target_id_index]
                    ReID_features = input_result[3]
                    ReID_feature = ReID_features[target_id_index]
                    img0 = input_result[4]
                    I_h, I_w, _ = img0.shape
                    x1, y1, w, h = box
                    intbox = tuple(map(int, (max(0, x1), max(0, y1), min(x1 + w, I_w), min(y1 + h, I_h))))
                    sub_img = img0[intbox[1]:intbox[3], intbox[0]:intbox[2]]

                    '''队列的首项是终极目标，用于校准，不用于后续的坐标转换计算'''
                    frames_time.append(None)
                    bottom_center_point_list.append(None)
                    # sub_imgs 和 ReID_feature_list 在后续直接用于计算，因此不需要与 frames_time 保持长度上的一致
                    sub_imgs.append(sub_img)
                    ReID_feature_list.append(ReID_feature)

                    if self.vis == True:
                        vis_dir_ = os.path.join(self.vis_path, '{}'.format(action_index), 'tracking')
                        makedir_v1(vis_dir_)

                        self.vis_target_frame(input_result, target_id, reference_point, new_reference_point,vis_dir_)

                    break
            # 如果前中后三帧都没有检测到，那就说明这个动作区分不开了。 放弃了。
            if target_id == None:
                # 目标不存在
                self.Output_Q.put((False,action_index,[]))
                continue

            # 对所有结果进行筛选，选出和目标人物相同ID的。
            for r_index, result in enumerate(results):

                frame_id = result[0]
                time = start_time + frame_id / frame_rate # 这帧画面对应的时间（相对于开球时间）
                bboxes = result[1]
                # ids = np.arryy(result[2])
                ids = result[2]
                ReID_features = result[3]
                img0 = result[4]
                I_h, I_w, _ = img0.shape

                # 找出复合target id的那一个bbox。 每一帧最多存在一个复合要求的box
                # 有可能有几帧是不存在的。因此需要标注时间，或者相对帧数
                if target_id not in ids:
                    # 需要记录每个sub_imgs所对应的时间。
                    # 要保证时间连续，就算没有信号，也需要添加在其中，
                    # 各个参数之间的长度需要保持一致
                    frames_time.append(time)
                    bottom_center_point_list.append([])
                    continue
                else:
                    id_index = ids.index(target_id)
                    box = bboxes[id_index]
                    ReID_feature = ReID_features[id_index]

                    x1, y1, w, h = box
                    intbox = tuple(map(int, (max(0, x1), max(0, y1), min(x1 + w, I_w), min(y1 + h, I_h))))
                    # print(intbox)
                    sub_img = img0[intbox[1]:intbox[3], intbox[0]:intbox[2]]

                    # 底部中心的坐标从相对于截图的位置，还原到相对于原图的位置。
                    bottom_center_point = (x1 + 0.5 * w + Left_Up_points[0], y1 + h + Left_Up_points[1])

                    # 需要记录每个bottom_center_point所对应的时间。
                    # frames_time 和 bottom_center_point 需要在长度上保持一致
                    frames_time.append(time)
                    bottom_center_point_list.append(bottom_center_point)

                    # sub_imgs 和 ReID_feature_list 在后续直接用于计算，因此不需要与 frames_time 保持长度上的一致
                    sub_imgs.append(sub_img)
                    ReID_feature_list.append(ReID_feature)

                    if self.vis ==True:
                        img_vis = np.copy(img0)
                        cv2.rectangle(img_vis,(intbox[0],intbox[1]),(intbox[2],intbox[3]),(255,255,0), thickness=2)
                        cv2.imwrite(os.path.join(vis_dir_,'{}.jpg'.format(r_index)),img_vis)
                        # cv2.imwrite(os.path.join(vis_dir_,'{}.jpg'.format(r_index)),img_vis)

            self.Output_Q.put((True,action_index,[frames_time,sub_imgs,ReID_feature_list,bottom_center_point_list]))
            if self.save_results == True:
                self.save_intermediate_resutls(action_index,frames_time,sub_imgs, ReID_feature_list,bottom_center_point_list)

            self.logger.log(21, 'FMLoader.PostProcess() action {} consums {}s'.format(action_index,self.PostProcess_timer.toc()))
            # show_memory_info('action _ {}, {}'.format(action_index, 'Before del results '))
            # print('action index {} sys.getrefcount(results)'.format(action_index),sys.getrefcount(results))
            # del results
            # show_memory_info('action _ {}, {}'.format(action_index, 'After del results '))

        # self.logger.log(21, '-----------------------------Finished FMLoader.PostProcess() datalen = {}-----------------------------'.format(self.datalen))

    def get__(self):
        for action_index in range(self.S_Short_track, self.datalen):
            results = self.Output_Q.get()
            self.logger.log(21, 'FMLoader.get__() action {} '.format(action_index))



    def save_intermediate_resutls(self,action_index,frames_time,sub_imgs, ReID_feature_list,bottom_center_point_list):
        '''将每一次计算的结果保存下来。'''
        intermediate_resutls_path = os.path.join(self.intermediate_results_dir,'{}'.format(action_index))
        os.makedirs(intermediate_resutls_path,exist_ok=True)
        # 保存 ReID
        ReID_feature_list = np.array(ReID_feature_list)
        np.save(os.path.join(intermediate_resutls_path,'{}_ReID_feature_list.npy'.format(action_index)),ReID_feature_list)
        # 保存图片
        for img_index in range(len(sub_imgs)):
            cv2.imwrite(os.path.join(intermediate_resutls_path,'{}.jpg'.format(img_index)),sub_imgs[img_index])
        # 保存 frames_time 和 bottom_center_point_list
        with open(os.path.join(intermediate_resutls_path,'{}_frames_time_and_bottom_center_point_list.json'.format(action_index)),'w') as f:
            results = {'frames_time' : frames_time, 'bottom_center_point_list':bottom_center_point_list}
            json.dump(results,f)

    def load_intermediate_resutls(self,action_index):
        '''将中间结果读取出来。'''
        intermediate_resutls_path = os.path.join(self.intermediate_results_dir,'{}'.format(action_index))

        ReID_feature_list = np.load(os.path.join(intermediate_resutls_path,'{}_ReID_feature_list.npy'.format(action_index)))
        ReID_feature_list = [ _ for _ in ReID_feature_list ] # 转换为我们需要的格式

        # 把这个文件夹下的图片名称读出来。
        sub_imgs_names = [ img_name for img_name in os.listdir(intermediate_resutls_path) if img_name.split('.')[-1] == 'jpg' ]
        # 把图片名字按升序排列
        sub_imgs_names = sorted(sub_imgs_names, key=lambda img_index : int(img_index.split('.')[0]))
        sub_imgs = []
        for img_name in sub_imgs_names:
            sub_img = cv2.imread(os.path.join(intermediate_resutls_path,img_name))
            sub_imgs.append(sub_img)
        with open(os.path.join(intermediate_resutls_path, '{}_frames_time_and_bottom_center_point_list.json'.format(action_index)),'r') as f:
            results = json.load(f)
            frames_time = results['frames_time']
            bottom_center_point_list = results['bottom_center_point_list']

        return action_index,[frames_time,sub_imgs,ReID_feature_list,bottom_center_point_list]

    def read(self):
        # return next frame in the queue
        return self.Output_Q.get()

    def len(self):
        # return queue len
        return self.Output_Q.qsize()

    def vis_target_frame(self,input_result,target_id,reference_point,new_reference_point,vis_dir_):
        '''
        将目标帧的效果画出来。
        '''
        bboxes = input_result[1]
        ids = input_result[2]
        img0 = input_result[4]
        img_vis = np.copy(img0)
        id_index = ids.index(target_id)
        box = bboxes[id_index]
        x1, y1, w, h = box
        I_h, I_w, _ = img0.shape
        intbox = tuple(map(int, (max(0, x1), max(0, y1), min(x1 + w, I_w), min(y1 + h, I_h))))
        cv2.rectangle(img_vis, (intbox[0], intbox[1]), (intbox[2], intbox[3]), (255, 255, 0), thickness=2) # 追踪效果和转换后的点为黄色
        cv2.circle(img_vis, (int(new_reference_point[0]), int(new_reference_point[1])), radius=5, color=(25, 255, 0),thickness=-1)
        cv2.circle(img_vis, (int(reference_point[0]), int(reference_point[1])), radius=5, color=(0, 255, 0), thickness=-1) # 原始点为红色

        cv2.imwrite(os.path.join(vis_dir_, 'target_id.jpg'),img0)