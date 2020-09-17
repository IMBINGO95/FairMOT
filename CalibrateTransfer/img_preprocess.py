import threading, queue, time
from queue import Queue
from threading import Thread, currentThread
from concurrent.futures import ThreadPoolExecutor

import os
from CalibrateTransfer.img_operation import ScreenSHot_batch
from CalibrateTransfer.data_preprocess import write_data_to_json_file,read_data_from_json_file,make_dir,read_subdata,read_stack_data
from CalibrateTransfer.cv_transfer import transform_2d_to_3d,object_To_pixel,updata_img_point
from CalibrateTransfer.img_operation import GenerateRect
from FairMot.lib.tracker.multitracker import JDETracker, create_JDETracker_model

import numpy as np
import torch.utils.data as data
import torch
import torch.multiprocessing as mp

from utils.sort_by_point import sort_by_point

import logging
from utils.log import Log
from utils.timer import Timer
from utils.dir_related_operation import makedir_v1
import cv2

class SubImgDetect(data.Dataset):  # for sub img detection
    def __init__(self,video, video_time, rect, Output_size, InQueue , img_size=(1088, 608), ):

        self.width, self.height = img_size[0] , img_size[1] # 网络输入的Feature Map的大小
        [self.vw, self.vh] = Output_size # 输入图片的大小
        [self.w, self.h] = Output_size # 可视化的图片的大小
        self.rect = rect # 对应的目标区域 [x_l,y_l,x_r,y_r]
        self.count = 0
        self.InQueue = InQueue
        # self.vn = 2 *multiple * self.frame_rate + 1
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        # Read image
        res, img0 = self.InQueue.get()
        assert res is False, 'Failed to load frame {:d}'.format(self.count)

        # Normalize RGB
        img = img0[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        self.count += 1
        if self.count == len(self):# 结束迭代的标志 输入队列空了,且结束标志为True
            raise StopIteration

        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of files

class Calibrate_transfer():
    def __init__(self,opt, detector_opt, Tracker_output_queue,C_T_output_queue,S_Coordinate_transfer,S_Pose_Estimate, vis=False, save_results=False ,queueSize=1024):

        self.logger = Log(__name__, 'Calibrate_transfer' ).getlog()

        self.opt = opt
        self.dir_name = opt.dir_name
        self.root_path = os.path.join(opt.data_root, '{}'.format(opt.dir_name))
        # logger.info('目标文件夹是{}'.format(self.root_path))
        self.file_name = opt.file_name
        # 本来就是要载入两次视频，分开读亦可以
        self.Videoparameters, \
        self.setting_parameter, \
        self.action_datas, \
        self.channel_list, \
        self.parameter = read_data_from_json_file(self.root_path, self.file_name, self.opt)

        self.datalen = len(self.action_datas)

        self.detector_opt = detector_opt  # 用来设置追踪器参数的。
        self.logger.info('Creating model...')
        self.detector_model = create_JDETracker_model(self.detector_opt)
        self.detector = JDETracker(self.detector_opt, self.detector_model)  # What is JDE Tracker? 把这个tracker 当detector用

        self.input_Q = Tracker_output_queue  # 追踪数据的整体输入
        self.PreProcess_Q = Queue(maxsize=queueSize) # 在目标检测前，对左边转换后的截图进行预处理
        self.tracking_Q = Queue(maxsize=queueSize)
        self.detecions_Q = Queue(maxsize=queueSize)
        self.output_Q = C_T_output_queue

        self.vis = vis
        if self.vis == True:
            self.vis_path = os.path.join(self.root_path, 'vis')
            os.makedirs(self.vis_path, exist_ok=True)


        self.S_Coordinate_transfer = S_Coordinate_transfer
        self.S_Pose_Estimate = S_Pose_Estimate
        self.save_results = save_results
        if self.save_results == True:
            self.intermediate_results_dir = os.path.join(self.root_path, 'intermediate_results','Calibrate_transfer')
            os.makedirs(self.intermediate_results_dir, exist_ok=True)


    def Read_From_Cache(self):
        '''
        从文件把之前计算过的结果提取出来
        '''
        from utils.index_operation import get_index

        self.logger.debug('The pid of Calibrate_transfer.Read_From_Cache() : {}'.format(os.getpid()))
        self.logger.debug('The thread of Calibrate_transfer.Read_From_Cache() : {}'.format(currentThread()))

        cache_index = get_index(self.intermediate_results_dir)
        # 只需读取有用的部分即可。
        action_index = self.S_Pose_Estimate
        for action_index in range(self.S_Pose_Estimate,self.S_Coordinate_transfer):

            if action_index not in cache_index:
                # cache 中没有保存说明 此动作本身是False
                self.output_Q.put((False,(action_index, [], [],[],[])))

            else:
                # 从文件夹中读取出该动作对应的计算结果。
                _, sub_img_tracking,ReID_features_tracking,sub_imgs_detection,ReID_features_detection  = self.load_intermediate_resutls(action_index)
                self.output_Q.put((True, (action_index, sub_img_tracking ,ReID_features_tracking,sub_imgs_detection,ReID_features_detection)))

        self.logger.log(22, ' Calibrate_transfer loads action {} from Cache file '.format(action_index))

    def update_(self):
        self.t_update = Thread(target=self.update, args=())
        self.t_update.daemon = True
        self.t_update.start()

    def update(self):
        '''
        将一个视角下的所有图片转换到其他视角下。
        '''
        self.logger.debug('The pid of Calibrate_transfer.update() : {}'.format(os.getpid()))
        self.logger.debug('The thread of Calibrate_transfer.update() : {}'.format(currentThread()))
        update_timer = Timer()
        sub_img_generate_timer = Timer()
        for action_index in range(self.S_Coordinate_transfer,self.datalen):

            update_timer.tic() # 开始计时
            self.logger.debug('update() ======================================== action {}'.format(action_index))
            Flag, input_index ,tracking_results = self.input_Q.get()
            if input_index != action_index:
                self.logger.log(31,'---——————————————————————————————————index does match')
                raise Exception('Calibrate_transfer.update action_index_update {} != input_index {} '.format(action_index, input_index))

            if Flag == False:
                # Flag == False 的话，直接就不要了
                self.tracking_Q.put((False, (action_index, [], [])))
                self.PreProcess_Q.put((False, (action_index, [])))
                continue

            frames_time, sub_imgs, ReID_feature_list, img_points = tracking_results
            # 分为 追踪结果和 对 每一帧追踪进行坐标转换后得到的检测结果
            # 这里先将追踪结果存入队列中。
            self.tracking_Q.put((True,(action_index, sub_imgs, ReID_feature_list)))

            channel,action_time,img_point,video_parameter = read_subdata(self.action_datas[action_index], self.Videoparameters)
            calibrateParameter = video_parameter['CalibrateParameter']

            # 将追踪结果对应的像素坐标转换成世界坐标
            '''队列的首项是终极目标，用于校准，不用于后续的坐标转换计算'''
            '''因此，直接从第二项开始'''
            world_points = []
            start_time = frames_time[1] # 追踪序列开始的时间，这里的时间是相对于开球时间
            for p_index in range(1, len(img_points)):
                img_point = img_points[p_index]
                # 输入的是连续的轨迹，因为检测原因，可能有诺干帧是没有img_points，长度因此为0
                if len(img_point) == 0:
                    world_points.append(None)
                else:
                    world_point = transform_2d_to_3d(img_point, calibrateParameter.cameraMatrix, calibrateParameter.distCoeffs,
                                                     calibrateParameter.rotation_vector,
                                                     calibrateParameter.translation_vector, world_z=0)

                    world_point = np.reshape(world_point, [1, 3])
                    world_points.append(world_point)

            # 将世界坐标转换到其他的视角下，并且 截图+detection\
            # print('len(world_points) : ', len(world_points))
            sub_img_generate_timer.tic()
            self.sub_img_generate_multi_thread(channel,action_index,world_points,start_time)
            # self.logger.info('Calibrate_transfer.sub_img_generate() action {} consums {}s'.format(action_index,sub_img_generate_timer.toc()))
            self.logger.log(22,'Calibrate_transfer.update() action {} consums {}s'.format(action_index,update_timer.toc()))

    def sub_img_generate(self,video_parameter, setting_parameter, world_points,start_time):
        '''
        基于世界坐标，生成其他视角下的像素坐标
        '''
        results = []
        video = video_parameter['video']
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # 将时间调整至追踪序列的开头，然后逐帧读取
        time = start_time + video_parameter['delta_t']  # action time need to add the delta time to calibrate the time between channels .
        video.set(cv2.CAP_PROP_POS_MSEC, round(1000 * time))

        for index in range(len(world_points)):
            _, img = video.read()
            world_point = world_points[index]

            if type(world_point) != np.ndarray:
                continue
            img_point_of_other = object_To_pixel(world_point, video_parameter['CalibrateParameter'])
            img_point_of_other = np.reshape(img_point_of_other, 2)

            Message = ScreenSHot_batch(img_point_of_other, img , setting_parameter, width, height)

            if Message[0] == True:
                # print('sub img of channel {} candidate {} exists'.format(other_channel,img_count_))
                image = Message[1]
                reference_point = Message[2]
                sub_imgs_bias = Message[3]
                results.append([index,image,reference_point,sub_imgs_bias])
            else:
                continue

        return results

    def sub_img_generate_multi_thread(self,channel,action_index, world_points,start_time):
        '''
        基于世界坐标，生成其他视角下的像素坐标
        '''
        results_all = []
        executor = ThreadPoolExecutor(max_workers=len(self.channel_list)-1)
        task_list = []

        for other_channel in self.channel_list:
            # 同一个视角，无需在生成截图
            if other_channel == channel:
                continue
            video_parameter = self.Videoparameters[other_channel]
            setting_parameter = self.setting_parameter

            task = executor.submit(self.sub_img_generate, video_parameter,setting_parameter,world_points,start_time)
            task_list.append(task)
        for task in task_list:
            while not task.done():
                time.sleep(1)
            results_all += task.result()

        if len(results_all) > 0 :
            self.PreProcess_Q.put((True,(action_index,results_all)))
        else:
            self.PreProcess_Q.put((False,(action_index,results_all)))


    def detect_(self):
        self.t_detect = Thread(target=self.detect, args=())
        self.t_detect.daemon = True
        self.t_detect.start()
    def detect(self):
        '''
        用检测其检测每一场图片中的人物
        '''
        detect_timer = Timer()
        self.logger.debug('The pid of Calibrate_transfer.detect() : {}'.format(os.getpid()))
        self.logger.debug('The thread of Calibrate_transfer.detect() : {}'.format(currentThread()))

        for action_index in range(self.S_Coordinate_transfer,self.datalen):

            self.logger.debug('Calibrate_transfer.Detection ------------action {} has been read '.format(action_index))
            Flag_PreProcess, (acton_index, Preprocess_results) = self.PreProcess_Q.get()
            detect_timer.tic()
            results = []

            if Flag_PreProcess == False:
                self.detecions_Q.put((False,(acton_index,results)))
                continue
            # 争取写成并行的
            for [index,img0,reference_point,sub_img_bias] in Preprocess_results:
                # Img preprocess before detection
                img = img0[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0
                # timer.tic()
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
                # detection using tracker kernel
                # dets = [xl, yl, w, h]
                [dets,id_feature] = self.detector.update_for_detection(blob, img0)

                if dets.shape[0] == 0 :
                    #截图中没有检测到人物, 继续
                    continue
                results.append([img0,dets,id_feature,reference_point,sub_img_bias])

            if len(results) > 0 :
                self.detecions_Q.put((True,(acton_index, results)))
            else:
                self.detecions_Q.put((False,(acton_index,results)))

            self.logger.log(22,'Calibrate_transfer.detect() action {} consums {}s'.format(action_index,detect_timer.toc()))

    def postProcess_(self):
        self.t_postProcess = Thread(target=self.postProcess, args=())
        self.t_postProcess.daemon = True
        self.t_postProcess.start()

    def postProcess(self):
        '''
        对检测完之后的结果进行后处理
        '''
        postProcess_timer = Timer()
        self.logger.debug('The pid of Calibrate_transfer.postProcess() : {}'.format(os.getpid()))
        self.logger.debug('The thread of Calibrate_transfer.postProcess() : {}'.format(currentThread()))

        for action_index in range(self.S_Coordinate_transfer,self.datalen):

            self.logger.debug('postProcess ------------action {} has been read '.format(action_index))

            Flag_detect, (acton_index_detection, results) = self.detecions_Q.get()
            Flag_tracking, (action_index_tracking,sub_imgs_tracking, ReID_features_tracking) = self.tracking_Q.get()

            postProcess_timer.tic()

            if Flag_detect == False or Flag_tracking == False:
                self.output_Q.put((False,(action_index, [], [],[],[] )))
                continue

            elif acton_index_detection != action_index or action_index_tracking != action_index:
                raise Exception('acton_index_detection {} != action_index_tracking {} '.format(acton_index_detection, action_index_tracking))

            if self.vis == True:
                vis_dir_ = os.path.join(self.vis_path, '{}'.format(action_index), 'Calibrate_transfer')
                makedir_v1(vis_dir_)

            # 把每个sub_box提取出来。
            sub_imgs_detection = []
            ReID_features_detection = []
            # 对所有结果进行筛选，选出和目标人物相同ID的。
            for r_index, [img0,dets,id_feature,reference_point,sub_img_bias] in enumerate(results):

                I_h, I_w, _ = img0.shape
                new_reference_point, target_id = sort_by_point([acton_index_detection,dets,False],reference_point,input_index='{}_{}'.format(action_index,r_index))
                if target_id == None:
                    '''根据reference_point来筛选框时，没有合适的框'''
                    if self.vis == True:
                        vis_img = np.copy(img0)
                        for cv2_index in range(int(dets.shape[0])):
                            box = dets[cv2_index].tolist()
                            x1, y1, w, h = box
                            c_intbox = tuple(map(int, (max(0, x1), max(0, y1), min(x1 + w, I_w), min(y1 + h, I_h))))
                            cv2.rectangle(vis_img, (c_intbox[0], c_intbox[1]), (c_intbox[2], c_intbox[3]), (255, 0, 0), thickness=2)
                        cv2.circle(vis_img, (int(reference_point[0]), int(reference_point[1])), radius=5, color=(0, 0, 255),thickness=-1)  # 原始点为红色
                        cv2.imwrite(os.path.join(vis_dir_, '{}.jpg'.format(r_index)), vis_img)
                        continue

                # print('dets.shape, target_id : ',dets.shape, target_id)
                target_bbox = dets[target_id]
                # print('target_bbox.shape : ', target_bbox.shape)
                target_bbox = target_bbox.tolist()
                # print('target_bbox : ', target_bbox)
                x1, y1, w, h = target_bbox
                # 目标区域
                intbox = tuple(map(int, (max(0, x1), max(0, y1), min(x1+w, I_w), min(y1+h, I_h))))
                sub_img = img0[intbox[1]:intbox[3], intbox[0]:intbox[2]]

                # ids = np.arryy(result[2])
                target_feature = id_feature[target_id]
                sub_imgs_detection.append(sub_img)
                ReID_features_detection.append(target_feature)

                if self.vis == True:
                    vis_img = np.copy(img0)
                    for cv2_index in range(int(dets.shape[0])):
                        box = dets[cv2_index].tolist()
                        x1, y1, w, h = box
                        c_intbox = tuple(map(int, (max(0, x1), max(0, y1), min(x1 + w, I_w), min(y1 + h, I_h))))
                        cv2.rectangle(vis_img, (c_intbox[0], c_intbox[1]), (c_intbox[2], c_intbox[3]), (255, 0, 0), thickness=2)
                    cv2.circle(vis_img, (int(reference_point[0]), int(reference_point[1])), radius=5, color=(0, 0, 255),thickness=-1)  # 原始点为红色
                    cv2.circle(vis_img, (int(new_reference_point[0]), int(new_reference_point[1])), radius=5, color=(0, 255, 255), thickness=-1)
                    cv2.rectangle(vis_img, (intbox[0], intbox[1]), (intbox[2], intbox[3]), (0, 255, 255), thickness=2)
                    cv2.imwrite(os.path.join(vis_dir_, '{}.jpg'.format(r_index)), vis_img)

            # 可以在此处加一个 ReID 模块 ，用于剔除劣质 sub_imgs
            sub_imgs = sub_imgs_detection + sub_imgs_tracking
            ReID_features = ReID_features_detection + ReID_features_tracking

            self.output_Q.put((True,(action_index, sub_imgs_tracking, ReID_features_tracking,sub_imgs_detection,ReID_features_detection)))
            # 保存中间结果
            if self.save_results==True:
                self.save_intermediate_resutls(action_index, sub_imgs, ReID_features,
                                               sub_imgs_detection, sub_imgs_tracking,
                                               ReID_features_detection,ReID_features_tracking)

            self.logger.log(22,'Calibrate_transfer.postProcess() action {} consums {}s'.format(action_index,postProcess_timer.toc()))
        # self.logger.log(22, '-----------------------------Finished Calibrate_transfer.postProcess() datalen = {}-----------------------------'.format(self.datalen))


    def save_intermediate_resutls(self,action_index, sub_imgs, ReID_features,
                                  sub_imgs_detection, sub_imgs_tracking,
                                  ReID_features_detection,ReID_features_tracking):
        '''将每一次计算的结果保存下来。'''
        intermediate_resutls_path = os.path.join(self.intermediate_results_dir,'{}'.format(action_index))
        os.makedirs(intermediate_resutls_path,exist_ok=True)
        ReID_features = np.array(ReID_features)
        np.save(os.path.join(intermediate_resutls_path,'{}_ReID_features.npy'.format(action_index)),ReID_features)
        for img_index in range(len(sub_imgs)):
            cv2.imwrite(os.path.join(intermediate_resutls_path,'{}.jpg'.format(img_index)),sub_imgs[img_index])

        # 保存tracking部分的img和feature
        intermediate_resutls_path_tracking = os.path.join(self.intermediate_results_dir,'{}/tracking'.format(action_index))
        os.makedirs(intermediate_resutls_path_tracking, exist_ok=True)
        ReID_features_tracking = np.array(ReID_features_tracking)
        np.save(
            os.path.join(intermediate_resutls_path_tracking, '{}_ReID_features_tracking.npy'.format(action_index)),
            ReID_features_tracking)
        for img_index_tracking in range(len(sub_imgs_tracking)):
            cv2.imwrite(os.path.join(intermediate_resutls_path_tracking, '{}.jpg'.format(img_index_tracking)),
                        sub_imgs_tracking[img_index_tracking])

        # 保存detection部分的img和feature
        intermediate_resutls_path_detection = os.path.join(self.intermediate_results_dir, '{}/detection'.format(action_index))
        os.makedirs(intermediate_resutls_path_detection, exist_ok=True)
        ReID_features_detection = np.array(ReID_features_detection)
        np.save(
            os.path.join(intermediate_resutls_path_detection, '{}_ReID_features_detection.npy'.format(action_index)), ReID_features_detection)
        for img_index_detection in range(len(sub_imgs_detection)):
            cv2.imwrite(os.path.join(intermediate_resutls_path_detection, '{}.jpg'.format(img_index_detection)), sub_imgs_detection[img_index_detection])

    def load_intermediate_resutls(self,action_index):
        '''将中间结果读取出来'''
        intermediate_resutls_path = os.path.join(self.intermediate_results_dir,'{}'.format(action_index))

        ReID_features = np.load(os.path.join(intermediate_resutls_path,'{}_ReID_features.npy'.format(action_index)))
        ReID_features = [ _ for _ in ReID_features ] # 转换为我们需要的格式

        # 把这个文件夹下的图片名称读出来。
        sub_imgs_names = [ img_name for img_name in os.listdir(intermediate_resutls_path) if img_name.split('.')[-1] == 'jpg' ]
        # 把图片名字按升序排列
        sub_imgs_names = sorted(sub_imgs_names, key=lambda img_index : int(img_index.split('.')[0]))
        sub_imgs = []
        for img_name in sub_imgs_names:
            sub_img = cv2.imread(os.path.join(intermediate_resutls_path,img_name))
            sub_imgs.append(sub_img)


        # 读取追踪部分
        intermediate_resutls_path_tracking = os.path.join(intermediate_resutls_path, 'tracking')
        ReID_features_tracking = np.load(os.path.join(intermediate_resutls_path_tracking, '{}_ReID_features_tracking.npy'.format(action_index)))
        ReID_features_tracking = [_ for _ in ReID_features_tracking]  # 转换为我们需要的格式

        # 把这个文件夹下的图片名称读出来。
        sub_imgs_names_tracking = [img_name_tracking for img_name_tracking in
                                    os.listdir(intermediate_resutls_path_tracking) if
                                    img_name_tracking.split('.')[-1] == 'jpg']
        # 把图片名字按升序排列
        sub_imgs_names_tracking = sorted(sub_imgs_names_tracking, key=lambda img_index: int(img_index.split('.')[0]))
        sub_imgs_tracking = []
        for img_name_tracking in sub_imgs_names_tracking:
            sub_img_tracking = cv2.imread(os.path.join(intermediate_resutls_path_tracking, img_name_tracking))
            sub_imgs_tracking.append(sub_img_tracking)

        # 读取 坐标转换部分
        intermediate_resutls_path_detection = os.path.join(intermediate_resutls_path,'detection')
        ReID_features_detection = np.load(os.path.join(intermediate_resutls_path_detection, '{}_ReID_features_detection.npy'.format(action_index)))
        ReID_features_detection = [_ for _ in ReID_features_detection]  # 转换为我们需要的格式

        # 把这个文件夹下的图片名称读出来。
        sub_imgs_names_detection = [img_name_detection for img_name_detection in os.listdir(intermediate_resutls_path_detection) if
                                    img_name_detection.split('.')[-1] == 'jpg']
        # 把图片名字按升序排列
        sub_imgs_names_detection = sorted(sub_imgs_names_detection, key=lambda img_index: int(img_index.split('.')[0]))
        sub_imgs_detection = []
        for img_name_detection in sub_imgs_names_detection:
            sub_img_detection = cv2.imread(os.path.join(intermediate_resutls_path_detection, img_name_detection))
            sub_imgs_detection.append(sub_img_detection)

        return action_index,sub_imgs_tracking,ReID_features_tracking,sub_imgs_detection,ReID_features_detection




        
        


if __name__ == "__main__":
    from opt import opt
    from FairMot.lib.opts import opts
    from CalibrateTransfer.img_operation import ScreenSHot

    detector_opt = opts().init()

    queueSize = 1000
    Tracker_output_queue = Queue(1000)
    dir_name = opt.dir_name
    root_path = os.path.join(opt.data_root, '{}'.format(dir_name))
    file_name = opt.file_name
    Videoparameters, \
    setting_parameter, \
    action_datas, \
    channel_list, \
    parameter = read_data_from_json_file(root_path, file_name, opt)
    vis_path = os.path.join(root_path, 'vis')
    os.makedirs(vis_path, exist_ok=True)

    multi = 10
    C_T_output_queue = Queue(queueSize)
    transfer = Calibrate_transfer(opt, detector_opt, Tracker_output_queue, C_T_output_queue, vis=True, queueSize=1024)
    transfer.update_()
    transfer.detect_()
    transfer.postProcess_()

    for index in range(len(action_datas)):
        channel,action_time,img_point,video_parameter = read_subdata(action_datas[index],Videoparameters)

        Message = ScreenSHot(img_point, action_time=action_time, video_parameter=video_parameter,
                             setting_parameter=setting_parameter)

        if Message[0] == True:
            count = 1
            # 根据操作员点击的点，进行区域截图。
            img0, reference_point, sub_img_bias = Message[1], Message[2], Message[3]
            cv2.circle(img0, (int(reference_point[0]), int(reference_point[1])), radius=5, color=(0, 255, 0),
                       thickness=-1)  # 原始点为红色
            vis_dir_ = os.path.join(vis_path,'{}'.format(index))
            os.makedirs(vis_dir_,exist_ok=True)
            cv2.imwrite(os.path.join(vis_dir_,'target.jpg'),img0)
            # detect bboxes and calculate the scores of each bbox

        img_points = [img_point] * multi
        Tracker_output_queue.put((True,index,[[action_time for i in range(multi)], [], [], img_points]))

    transfer.t_update.join()
    transfer.t_detect.join()
    transfer.t_postProcess.join()


