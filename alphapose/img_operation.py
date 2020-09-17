import threading, queue, time
from queue import Queue
from threading import Thread, currentThread

import logging
from utils.log import Log
from utils.timer import Timer
from utils.dir_related_operation import makedir_v1
import os
import cv2
import numpy as np
import torch
import json
import shutil

from CalibrateTransfer.data_preprocess import write_data_to_json_file,read_data_from_json_file_v2,make_dir,read_subdata,read_stack_data

from ReID_model.modeling import ReID_Model
from utils_BINGO.K_Means import k_means
from ReID_model.utils.dataset_loader import transform_imgs,ReID_imgs_loadv1
from ReID_model.imgs_sort_by_ReID import imgs_sorted_by_ReID

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.presets import SimpleTransform
from alphapose.utils.detector import DetectedImgsLoader
from alphapose.utils.pPose_nms import write_json
from alphapose.utils.transforms import flip, flip_heatmap, heatmap_to_coord_simple
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.writer import DataWriter, heat_to_map

def build_poser(pose_opt,gpus_id='0'):
    # Load pose model
    pose_model = builder.build_sppe(pose_opt.MODEL, preset_cfg=pose_opt.DATA_PRESET)
    pose_model.cuda()
    # gpus_id = [int(i) for i in gpus_id.split(',')] if torch.cuda.device_count() >= 1 else [-1]
    # device = torch.device("cuda:" + str(gpus_id[0]) if gpus_id[0] >= 0 else "cpu")

    # print(f'Loading alphapose model from {pose_opt.MODEL.checkpoint}...')
    pose_model.load_state_dict(torch.load(pose_opt.MODEL.checkpoint))

    # if len(gpus_id) > 1:
    #     pose_model = torch.nn.DataParallel(pose_model, device_ids=gpus_id).to(device)
    # else:
    #     pose_model.to(device)
    pose_model.eval()

    return pose_model

class Alphapose():
    def __init__(self,opt, pose_opt, ReIDCfg, C_T_output_queue,Pose_output_queue, S_Pose_Estimate, S_Number_Predict, vis=False,save_results=False, queueSize=1024):

        self.opt = opt
        self.dir_name = opt.dir_name
        self.root_path = os.path.join(opt.data_root, '{}'.format(opt.dir_name))
        # logger.info('目标文件夹是{}'.format(self.root_path))
        self.file_name = opt.file_name

        self.Videoparameters, \
        self.setting_parameter, \
        self.action_datas, \
        self.channel_list, \
        self.parameter = read_data_from_json_file_v2(self.root_path, self.file_name, self.opt) # 视频是否需要再次读入呢？ 是否暂用资源


        self.datalen = len(self.action_datas)

        # 加载 poser
        self.device = torch.device('cuda')
        self.batchSize = 4
        self.ReID_BatchSize = 50
        self.gpus = opt.gpus
        self.pose_model = build_poser(pose_opt,self.gpus)

        # 加载 ReID 模型
        self.ReIDCfg = ReIDCfg
        self.ReID = ReID_Model(self.ReIDCfg)
        self.ReID.cuda()


        # ReID 模型参数
        self.distance_threshold = 1
        self.height_threshold = 95
        self.width_threshold = 40

        self._input_size = pose_opt.DATA_PRESET.IMAGE_SIZE
        self._output_size = pose_opt.DATA_PRESET.HEATMAP_SIZE
        self._sigma = pose_opt.DATA_PRESET.SIGMA
        self.aspect_ratio = 0.45


        if pose_opt.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                self, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)


        self.input_Q = C_T_output_queue  # 追踪数据的整体输入
        self.Posing_Q = Queue(maxsize=queueSize) #在骨骼关键点检测前，对左边转换后的截图进行预处理
        self.PostProcess_Q = Queue(maxsize=queueSize) # 在骨骼关键点检测前，对左边转换后的截图进行预处理
        self.output_Q = Pose_output_queue

        self.vis = vis
        if self.vis == True:
            self.vis_path = os.path.join(self.root_path, 'vis')
            os.makedirs(self.vis_path, exist_ok=True)

        self.save_results = save_results
        self.S_Pose_Estimate = S_Pose_Estimate
        self.S_Number_Predict = S_Number_Predict

        if self.save_results == True:
            self.intermediate_results_dir = os.path.join(self.root_path, 'intermediate_results','Alphapose')
            os.makedirs(self.intermediate_results_dir, exist_ok=True)

        self.logger = Log(__name__, 'Alphapose' ).getlog()

    def Read_From_Cache(self):
        '''
        从文件把之前计算过的结果提取出来
        '''
        from utils.index_operation import get_index

        self.logger.debug('The pid of Alphapose.Read_From_Cache() : {}'.format(os.getpid()))
        self.logger.debug('The thread of Alphapose.Read_From_Cache() : {}'.format(currentThread()))

        cache_index = get_index(self.intermediate_results_dir)
        # 只需读取有用的部分即可。
        action_index = self.S_Number_Predict
        for action_index in range(self.S_Number_Predict,self.S_Pose_Estimate):

            if action_index not in cache_index:
                # cache 中没有保存说明 此动作本身是False
                self.output_Q.put((False, [action_index]))

            else:
                # 从文件夹中读取出该动作对应的计算结果。
                _, sub_imgs_out, target_regions = self.load_intermediate_resutls(action_index)
                self.output_Q.put((True, [action_index,sub_imgs_out,target_regions]))

        self.logger.log(23, ' Alphapose loads action {} from Cache file '.format(action_index))


    def posing_preprocess_(self):
        self.t_posing_preprocess = Thread(target=self.posing_preprocess, args=())
        self.t_posing_preprocess.daemon = True
        self.t_posing_preprocess.start()

    def posing_preprocess(self):
        # 预处理
        self.logger.debug('The pid of Alphapose.posing_preprocess() : {}'.format(os.getpid()))
        self.logger.debug('The thread of Alphapose.posing_preprocess() : {}'.format(currentThread()))
        posing_preprocess_timer = Timer()
        for action_index in range(self.S_Pose_Estimate, self.datalen):

            self.logger.debug('alphapose.posing_preprocess() ======================================== action {}'.format(action_index))
            Flag_PreProcess, (input_index, sub_imgs_tracking, ReID_features_tracking,sub_imgs_detection,ReID_features_detection) = self.input_Q.get()
            if input_index != action_index:
                self.logger.log(31, '---——————————————————————————————————index does match')
                raise Exception('Alphapose.update action_index_update {} != input_index {} '.format(action_index, input_index))

            if Flag_PreProcess == False:
                self.Posing_Q.put((False,[]))
                continue
            else:
                # 开始计时
                posing_preprocess_timer.tic()

                inps = []
                cropped_boxes = []

                # 通过 ReID 特征剔除一部分。

                sub_imgs = self.imgs_sorted_by_ReID(sub_imgs_tracking,sub_imgs_detection,action_index)

                if len(sub_imgs) == 0 :
                    # 被筛选后的图片序列为0，则跳过。
                    self.Posing_Q.put((False, []))
                    continue

                for imgs_index in range(len(sub_imgs)):
                    orig_img = sub_imgs[imgs_index]
                    height, width, _ = orig_img.shape
                    box = [0, 0, width - 1, height - 1]
                    inp, cropped_box = self.transformation.test_transform(orig_img, box)
                    inps.append(inp)
                    cropped_boxes.append(cropped_box)

                inps_ = torch.stack(inps,dim=0)
                self.Posing_Q.put((True,(sub_imgs,inps_,cropped_boxes)))
                self.logger.log(23, 'alphapose.posing_preprocess() action {} consums {}s'.format(action_index,
                                                                                                 posing_preprocess_timer.toc()))

    def posing_detect_(self):
        self.t_posing_detect = Thread(target=self.posing_detect, args=())
        self.t_posing_detect.daemon = True
        self.t_posing_detect.start()

    def posing_detect(self):
        posing_detect_timer = Timer()

        for action_index in range(self.S_Pose_Estimate, self.datalen):
            self.logger.debug('posing_detect ------------action {} has been read '.format(action_index))
            Flag_Posing_detect, preprocess_results = self.Posing_Q.get()

            if Flag_Posing_detect == False:
                self.PostProcess_Q.put((False,[]))
                continue
            else:
                posing_detect_timer.tic()
                sub_imgs, inps_ , cropped_boxes = preprocess_results
                inps = inps_.to(self.device)
                inps_len = inps_.size(0)
                leftover = 0
                if (inps_len) % self.batchSize:
                    leftover = 1
                num_batches = inps_len // self.batchSize + leftover
                keypoints_all = []
                for j in range(num_batches):
                    inps_j = inps[j * self.batchSize : min((j + 1) * self.batchSize, inps_len)]
                    sub_cropped_boxes = cropped_boxes[j * self.batchSize : min((j + 1) * self.batchSize, inps_len)]
                    # self.logger.log(23, ' j : {}, inps_j.size() '.format(j, inps_j.size()))
                    hm_j = self.pose_model(inps_j)
                    keypoints_several = self.heats_to_maps(hm_j, sub_cropped_boxes)
                    keypoints_all.extend(keypoints_several)

                self.PostProcess_Q.put((True,(keypoints_all,sub_imgs)))
                self.logger.log(23, 'alphapose.posing_detect() action {} consums {}s'.format(action_index,
                                                                                                 posing_detect_timer.toc()))

    def posing_postprocess_(self):
        self.t_posing_postprocess = Thread(target=self.posing_postprocess, args=())
        self.t_posing_postprocess.daemon = True
        self.t_posing_postprocess.start()

    def posing_postprocess(self):
        '''对骨骼关键节点的检测结果坐后处理，并通过 简单规则对结果进行以此初步筛选。'''
        pposing_postprocess_timer = Timer()
        for action_index in range(self.S_Pose_Estimate, self.datalen):
            self.logger.debug('posing_postprocess ------------action {} has been read '.format(action_index))

            Flag_posing_postprocess, posing_detect_resutls = self.PostProcess_Q.get()

            if Flag_posing_postprocess == False:
                self.output_Q.put((False,[action_index]))
                continue
            else:
                pposing_postprocess_timer.tic()
                keypoints_all,sub_imgs = posing_detect_resutls

                target_regions = []
                sub_imgs_out = []

                if self.vis == True:
                    vis_dir_positive = os.path.join(self.vis_path, '{}'.format(action_index), 'Alphapose_positive')
                    makedir_v1(vis_dir_positive)
                    vis_dir_negative = os.path.join(self.vis_path, '{}'.format(action_index), 'Alphapose_negative')
                    makedir_v1(vis_dir_negative)
                    Negative_num = 0
                    vis_dir_small_size = os.path.join(self.vis_path, '{}'.format(action_index), 'Alphapose_small_size')
                    makedir_v1(vis_dir_small_size)
                    small_size_num = 0
                    vis_dir_small_target = os.path.join(self.vis_path, '{}'.format(action_index), 'Alphapose_small_target')
                    makedir_v1(vis_dir_small_target)
                    small_target_num = 0

                Positive_num = 0
                for k_index in range(len(keypoints_all)):
                    # 对每一张关节点图做逐一处理
                    origin_img = sub_imgs[k_index]
                    height, width, _ = origin_img.shape

                    if height < self.height_threshold or width < self.width_threshold:
                        small_size_num += 1
                        if self.vis == True:
                            img_name = '{}.jpg'.format(k_index)
                            cv2.imwrite(os.path.join(vis_dir_small_size, img_name), origin_img)
                        continue
                    keypoints = keypoints_all[k_index]

                    # 这个判断标准和get_box的标准不一样。
                    # 用来判断是否背向的
                    l_x_max = max(keypoints[5 * 3], keypoints[11 * 3])
                    r_x_min = min(keypoints[6 * 3], keypoints[12 * 3])
                    t_y_max = max(keypoints[5 * 3 + 1], keypoints[6 * 3 + 1])
                    b_y_min = min(keypoints[11 * 3 + 1], keypoints[12 * 3 + 1])

                    if l_x_max < r_x_min and t_y_max < b_y_min:
                        '初步判断球员是否背向'
                        [xmin_old, xmax_old], [xmin, xmax, ymin, ymax] = self.get_box(keypoints, height, width, ratio=0.1,
                                                                                 expand_w_min=10)
                        # 计算上半身体长度
                        body_length = ymax - ymin
                        if body_length < 20:  # 130 和 60 应该来自 opt
                            small_target_num += 1
                            if self.vis == True:
                                img_name = '{}.jpg'.format(k_index)
                                cv2.imwrite(os.path.join(vis_dir_small_target, img_name), origin_img)
                            continue

                        # 计算肩宽、胯宽
                        Shoulder_width = keypoints[6 * 3] - keypoints[5 * 3]
                        Crotch_width = keypoints[12 * 3] - keypoints[11 * 3]

                        aspect_ratio = (max(Shoulder_width, Crotch_width)) / (body_length) # 计算比例
                        if aspect_ratio >= self.aspect_ratio:
                            # 如果这个比例合适，则送入号码检测
                            sub_imgs_out.append(origin_img)
                            target_regions.append([xmin, xmax, ymin, ymax])
                            Positive_num += 1 # 复合条件的 +1
                            if self.vis == True:
                                img_name = '{}.jpg'.format(k_index)
                                vis_img = np.copy(origin_img)
                                cv2.rectangle(vis_img, (xmin_old, ymin), (xmax_old, ymax), color=(255, 0, 0), thickness=1)
                                cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)
                                cv2.imwrite(os.path.join(vis_dir_positive, img_name), vis_img)
                    else:
                        Negative_num += 1
                        if self.vis == True:
                            img_name = '{}.jpg'.format(k_index)
                            cv2.imwrite(os.path.join(vis_dir_negative, img_name), origin_img)

                self.output_Q.put((True, [action_index, sub_imgs_out,target_regions ]))
                # 保存中间结果

                if self.save_results == True:
                    self.save_intermediate_resutls(action_index,sub_imgs_out,target_regions)
                # # 输出 日志
                # self.logger.log(23,'Positive_num {}, Negative_num {}, small_target_num {}, small_size_num {}, all {}'.format(
                #     Positive_num,
                #     Negative_num,
                #     small_target_num,
                #     small_size_num,
                #     len(keypoints_all)))

                self.logger.log(23, 'alphapose.posing_postprocess() action {} consums {}s Positive_num / All = {}/{}'.format(
                    action_index,
                    pposing_postprocess_timer.toc(),
                    Positive_num,
                    len(keypoints_all)))

    def save_intermediate_resutls(self,action_index,sub_imgs_out,target_regions):
        '''将每一次计算的结果保存下来。'''
        intermediate_resutls_path = os.path.join(self.intermediate_results_dir,'{}'.format(action_index))
        os.makedirs(intermediate_resutls_path,exist_ok=True)
        # 保存图片
        for img_index in range(len(sub_imgs_out)):
            cv2.imwrite(os.path.join(intermediate_resutls_path,'{}.jpg'.format(img_index)),sub_imgs_out[img_index])
        # 保存 target_regions
        with open(os.path.join(intermediate_resutls_path,'{}_target_regions.json'.format(action_index)),'w') as f:
            results = {'target_regions' : target_regions}
            json.dump(results,f)

    def load_intermediate_resutls(self,action_index):
        '''将中间结果读取出来。'''
        intermediate_resutls_path = os.path.join(self.intermediate_results_dir,'{}'.format(action_index))
        # 把这个文件夹下的图片名称读出来。
        sub_imgs_names = [ img_name for img_name in os.listdir(intermediate_resutls_path) if img_name.split('.')[-1] == 'jpg' ]
        # 把图片名字按升序排列
        sub_imgs_names = sorted(sub_imgs_names, key=lambda img_index : int(img_index.split('.')[0]))
        sub_imgs_out = []
        for img_name in sub_imgs_names:
            sub_img = cv2.imread(os.path.join(intermediate_resutls_path,img_name))
            sub_imgs_out.append(sub_img)
            # 保存 target_regions

        with open(os.path.join(intermediate_resutls_path, '{}_target_regions.json'.format(action_index)), 'r') as f:
            results = json.load(f)
            target_regions = results['target_regions']

        return action_index,sub_imgs_out,target_regions

    def heats_to_maps(self,hm_data,cropped_boxes):
        # 将 heatmap 转化成  keypoints 数组
        pred = hm_data.cpu().data.numpy()
        assert pred.ndim == 4

        keypoints_all = []
        for hms_index in range(hm_data.size(0)):
            pose_coord, pose_score = heatmap_to_coord_simple(pred[hms_index], cropped_boxes[hms_index])
            keypoints_single = []
            for n in range(pose_score.shape[0]):
                keypoints_single.append(float(pose_coord[n, 0]))
                keypoints_single.append(float(pose_coord[n, 1]))
                keypoints_single.append(float(pose_score[n]))
            keypoints_all.append(keypoints_single)

        return keypoints_all

    def get_box(self, keypoints, img_height, img_width, ratio=0.1, expand_w_min=10):
        '''这个get box 是用来获取球员的背部区域的'''
        xmin = min(keypoints[5 * 3], keypoints[11 * 3])
        xmax = max(keypoints[6 * 3], keypoints[12 * 3])
        ymin = min(keypoints[5 * 3 + 1], keypoints[6 * 3 + 1])
        ymax = max(keypoints[11 * 3 + 1], keypoints[12 * 3 + 1])

        return [int(round(xmin)), int(round(xmax))], self.expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height,
                                                                 ratio, expand_w_min)
    def expand_bbox(self, left, right, top, bottom, img_width, img_height,ratio = 0.1, expand_w_min = 10):
        '''
        以一定的ratio向左右外扩。 不向上向下扩展了。
        '''
        width = right - left
        height = bottom - top
         # expand ratio
        expand_w_min = max(ratio * width , expand_w_min) # 最小外扩 expand_w_min
        new_left = np.clip(left - expand_w_min, 0, img_width)
        new_right = np.clip(right + expand_w_min, 0, img_width)

        return [int(new_left), int(new_right), int(top), int(bottom)]

    def imgs_sorted_by_ReID(self,imgs_tracking,imgs_detection,action_index):
        '''通过ReID模型来筛选与目标特征相符的图片'''

        sub_imgs = []
        # 把追踪序列和目标人物进行对比，剔除后得到追踪序列的平均ReID特征值
        if len(imgs_tracking) == 0:
            # 如果追踪序列长度为0的话，那就没什么好处理的了，直接返回 空 就行。
            return sub_imgs
        else:
            imgs_tracking_index, distmat_tracking, output_feature = imgs_sorted_by_ReID(self.ReID, self.ReIDCfg, imgs_tracking,
                                                                                        distance_threshold=self.distance_threshold,
                                                                                        feat_norm='yes',
                                                                                        version=0,
                                                                                        batch_size=self.ReID_BatchSize)
            for P_index in imgs_tracking_index:
                sub_imgs.append(imgs_tracking[P_index])

            if len(imgs_detection) > 0:
                # 把追踪序列的平均ReID特征值和坐标转换序列对比，进行第二次筛选
                imgs_detection_index, distmat_detection, _ = imgs_sorted_by_ReID(self.ReID, self.ReIDCfg, imgs_detection,
                                                                                 distance_threshold=self.distance_threshold,
                                                                                 feat_norm='yes',
                                                                                 version=2,
                                                                                 input_features=output_feature,
                                                                                 batch_size=self.ReID_BatchSize)
                for P_index_detection in imgs_detection_index:
                    sub_imgs.append(imgs_detection[P_index_detection])

            if self.vis ==True:
                # 将追踪序列的sub_imgs 按ReID的分类结果保存
                Positive_dir = os.path.join(self.vis_path, '{}/ReID'.format(action_index))
                makedir_v1(Positive_dir)
                Negative_dir = os.path.join(self.vis_path, '{}/ReID/Negative'.format(action_index))

                for P_index, _ in enumerate(imgs_tracking):
                    distance = distmat_tracking[0, P_index]
                    if P_index in imgs_tracking_index:
                        cv2.imwrite(os.path.join(Positive_dir, '{}_{:3f}.jpg'.format(P_index, distance)), imgs_tracking[P_index])
                    else:
                        cv2.imwrite(os.path.join(Negative_dir, '{}_{:3f}.jpg'.format(P_index, distance)), imgs_tracking[P_index])

                # 将坐标转换后序列的sub_imgs 按ReID的分类结果保存
                Positive_dir_detection = os.path.join(self.vis_path, '{}/ReID/detection'.format(action_index))
                makedir_v1(Positive_dir_detection)
                Negative_dir_detection = os.path.join(self.vis_path, '{}/ReID/detection/Negative'.format(action_index))
                makedir_v1(Negative_dir_detection)
                for P_index_detection, _ in enumerate(imgs_detection):
                    distance = distmat_detection[0, P_index_detection]
                    if P_index_detection in imgs_detection_index:
                        cv2.imwrite(os.path.join(Positive_dir_detection, '{}_{:3f}.jpg'.format(P_index_detection, distance)),
                                    imgs_detection[P_index_detection])
                    else:
                        cv2.imwrite(os.path.join(Negative_dir_detection, '{}_{:3f}.jpg'.format(P_index_detection, distance)),
                                    imgs_detection[P_index_detection])

            return sub_imgs

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]





def generate_img_sequences(opt, C_T_output_queue):
    dir = '/datanew/hwb/data/SJN-210k/test/JPEGImages'
    imgs_list = os.listdir(dir)
    imgs = []
    for name in imgs_list:
        if name.split('.')[-1] == 'jpg':
            img = cv2.imread(os.path.join(dir,name))
            imgs.append(img)

    for index in range(1000):
        C_T_output_queue.put(True, (index, imgs, []))


if __name__ == '__main__':
    import torch.multiprocessing as mp

    from SoftWare_main import Pose_Estimate

    # 追踪器的参数
    from opt import opt
    Pose_opt = update_config(opt.Poser_cfg)

    queueSize = 1024
    cfg_file = '/datanew/hwb/FairMOT-master/alphapose/configs/coco/resnet/256x192_res50_lr1e-3_1x-simple.yaml'
    cfg = update_config(cfg_file)

    C_T_output_queue = mp.Queue(queueSize) # C_T : coordinate transfer.
    # 基于追踪数据，将追踪数据转换到其他的视角，并且生成相应的截图何ReID Features
    C_transfer = mp.Process(target=generate_img_sequences, args=(opt, C_T_output_queue))
    C_transfer.daemon = True
    C_transfer.start()
    Pose_output_queue = mp.Queue(queueSize)
    # 基于追踪数据，将追踪数据转换到其他的视角，并且生成相应的截图何ReID Features
    P_estimate = mp.Process(target=Pose_Estimate, args=(opt, Pose_opt, C_T_output_queue, Pose_output_queue))
    P_estimate.daemon = True
    P_estimate.start()

