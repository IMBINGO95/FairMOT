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
    def __init__(self, opt, ReIDCfg, Num_Pred_opt, Pose_output_queue, S_Number_Predict,
                 vis=False, save_results=False, queueSize=1024):

        self.opt = opt
        self.dir_name = opt.dir_name
        self.root_path = os.path.join(opt.data_root, '{}'.format(opt.dir_name))
        # logger.info('目标文件夹是{}'.format(self.root_path))
        self.file_name = opt.file_name

        self.file_save_name_before_Number_Rectify = 'Step1_'
        self.file_save_name_after_Number_Rectify = 'Step2_'


        # 本来就是要载入两次视频，分开读亦可以
        self.Videoparameters, \
        self.setting_parameter, \
        self.action_datas, \
        self.channel_list, \
        self.parameter = read_data_from_json_file_v2(self.root_path, self.file_name, self.opt)

        # 号码纠正器， 根据四官报告来修改参数
        self.Number_Rectifier = Number_Rectifier

        self.datalen = len(self.action_datas)
        self.batch_size = 60

        self.Num_Pred_opt = Num_Pred_opt  # 用来设置号码识别模型的参数。
        self.SVHN_predictor = load_in_Svhn_model(self.Num_Pred_opt)

        self.input_Q = Pose_output_queue  # 骨骼关键节点检测后的输入结果
        self.PreProcess_Q = Queue(maxsize=queueSize)  # 在号码识别前，对输入图片进行预处理。
        self.SVHN_Q = Queue(maxsize=queueSize)

        self.transform = transforms.Compose([
            transforms.Resize([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.vis = vis
        if self.vis == True:
            self.vis_path = os.path.join(self.root_path, 'vis')
            os.makedirs(self.vis_path, exist_ok=True)

        self.S_Number_Predict = S_Number_Predict
        self.S_Final = max( S_Number_Predict - 1, 0)
        self.height_threshold = 21
        self.width_threshold = 12
        self.save_results = save_results
        if self.save_results == True:
            self.intermediate_results_dir = os.path.join(self.root_path, 'intermediate_results', 'SVHN_Predict')
            os.makedirs(self.intermediate_results_dir, exist_ok=True)

        self.main_imgs_dir = os.path.join(self.root_path, 'intermediate_results', 'main_imgs')
        self.FMLoader_dir = os.path.join(self.root_path, 'intermediate_results', 'FMLoader')

        # 加载 ReID 模型
        self.ReIDCfg = ReIDCfg
        self.num_cls = 4 # 场上有几种类型的人

        self.logger = Log(__name__, 'SVHN_Predict').getlog()

    def Read_From_Cache(self):
        '''
        从文件把之前计算过的结果提取出来
        '''
        self.logger.debug( 'The pid of SVHN_Predict.Read_From_Cache() : {}'.format(os.getpid()))
        self.logger.debug( 'The thread of SVHN_Predict.Read_From_Cache() : {}'.format(currentThread()))
        self.load_intermediate_resutls(self.S_Final)
        self.logger.log(24, ' SVHN_Predict loads action {} from Cache file '.format(self.S_Final))

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
        for action_index in range(self.S_Number_Predict, self.datalen):
            PreProcess_timer.tic()  # 开始计时
            self.logger.debug('PreProcess() ======================================== action {}'.format(action_index))
            Flag, input_results = self.input_Q.get()

            if Flag == False:
                # Flag == False 的话，直接就不要了
                self.PreProcess_Q.put((False, (action_index, [])))
                continue
            #输入的数据有意义，可以接着处理
            [input_index, sub_imgs_out, target_regions] = input_results

            if input_index != action_index:
                self.logger.log(31, '---——————————————————————————————————index does match')
                raise Exception(
                    'SVHN_Predict.PreProcess action_index_update {} != input_index {} '.format(action_index,
                                                                                               input_index))
            # 对数据进行预处理。
            rectangle_imgs,original_imgs = self.img_pre_for_SVHN(sub_imgs_out,target_regions)

            if type(rectangle_imgs) != torch.Tensor:
                self.PreProcess_Q.put((False, (action_index, [])))
            else:
                self.PreProcess_Q.put((True, (action_index, rectangle_imgs, original_imgs)))

            # self.logger.info('Calibrate_transfer.sub_img_generate() action {} consums {}s'.format(action_index,sub_img_generate_timer.toc()))
            self.logger.log(24, 'SVHN_Predict.PreProcess() action {} consums {}s'.format(action_index, PreProcess_timer.toc()))


    def img_pre_for_SVHN(self,sub_imgs_out,target_regions):
        '''
        对需要 SVHN 的图片进行 数据预处理的 具体操作
        '''
        rectangle_imgs = []
        original_imgs = []
        for target_index in range(len(target_regions)) :
            sub_img = sub_imgs_out[target_index]
            [xmin, xmax, ymin, ymax] = target_regions[target_index]

            i_height, i_weight, i_channel = sub_img.shape
            crop_img = sub_img[max(ymin, 0):min(i_height, ymax), max(xmin, 0):min(xmax, i_weight)]
            h_i, w_i, _ = crop_img.shape
            if h_i < self.height_threshold or w_i < self.width_threshold:
                # 如果背部区域太小了，也要舍弃。
                continue
            crop_image = Image.fromarray(crop_img)
            crop_image = self.transform(crop_image)
            rectangle_imgs.append(crop_image)
            original_imgs.append(sub_img)
        # 如果都不符合条件的话。
        if len(rectangle_imgs) == 0:
            return None, None

        rectangle_imgs = torch.stack(rectangle_imgs, dim=0)
        return rectangle_imgs,original_imgs

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
        for action_index in range(self.S_Number_Predict, self.datalen):
            Predict_timer.tic() # 开始计时
            PreProcess_Flag, PreResults = self.PreProcess_Q.get()
            self.logger.debug('Predict() ======================================== action {}'.format(action_index))

            if PreProcess_Flag == False:
                # 输入的数据无意义
                preNum = -1
                self.action_datas[action_index]['predicted_nums'] = []

            else:
                # 输入的数据有意义， 读取数据
                _, rectangle_imgs,original_imgs = PreResults
                imgs_length = rectangle_imgs.size(0)
                leftover = 0
                if (imgs_length) % self.batch_size:
                    leftover = 1
                num_batches = imgs_length // self.batch_size + leftover

                if self.vis == True:
                    vis_dir = os.path.join(self.vis_path,'{}'.format(action_index),'SVHN_Predict')
                    makedir_v1(vis_dir)
                    vis_dir_0 = os.path.join(self.vis_path, '{}'.format(action_index), 'SVHN_Predict_Minus_one')
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
                            if self.vis == True:
                                cv2.imwrite(os.path.join(vis_dir_0, '{}_P{}.jpg'.format(num_batches*j + Num_i, Num)), original_imgs[Num_i])
                            continue
                        else:
                            continue

                        if self.vis == True:
                            cv2.imwrite(os.path.join(vis_dir, '{}_P{}.jpg'.format(num_batches*j + Num_i, Num)), original_imgs[Num_i])

                    NumsArray.extend(NumsArray_j)

                # 将数据保存下来
                self.action_datas[action_index]['predicted_nums'] = NumsArray

                if len(NumsArray) > 1:
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
            True_num = self.action_datas[action_index]['num']
            self.action_datas[action_index]['num'] = '{}'.format(preNum)
            self.logger.log(24, 'SVHN_Predict.Predict action {} consums {}s'.format(action_index, Predict_timer.toc()))
            self.logger.log(24,'action {} ====================== True num = {}, Predict num = {} ============='.format(
                action_index,True_num,preNum))

            if self.save_results == True:
                self.save_intermediate_resutls(action_index)

        self.logger.log(24, '-----------------------------Finished SVHN_Predict.Predict() datalen = {}-----------------------------'.format(self.datalen))

        # Finished 完成了所有的计算，保存最终结果,未进行号码矫正
        write_data_to_json_file(self.root_path, self.file_name, self.action_datas, self.parameter, file_save_name=self.file_save_name_before_Number_Rectify)

        # 根据四官报告修改最终结果
        self.action_datas = self.Number_Rectifier(os.path.join(self.root_path,self.file_save_name_before_Number_Rectify + self.file_name)).rectify()
        self.logger.log(24, 'Successfully Rectify numbers according to four officials report')
        write_data_to_json_file(self.root_path, self.file_name, self.action_datas, self.parameter, file_save_name=self.file_save_name_after_Number_Rectify)

        # 并根据ReID特征划分主图片。
        self.cluster_main_imgs()

    def save_intermediate_resutls(self, action_index):
        '''将每一次计算的结果保存下来。'''
        intermediate_resutls_path = os.path.join(self.intermediate_results_dir,'{}'.format(action_index))
        os.makedirs(intermediate_resutls_path,exist_ok=True)
        json_file = os.path.join(intermediate_resutls_path, '{}_action_data.json'.format(action_index))
        with open(json_file,'w') as f:
            json.dump(self.action_datas,f)

    def load_intermediate_resutls(self, action_index):
        '''将中间结果读取出来'''
        intermediate_resutls_path = os.path.join(self.intermediate_results_dir, '{}'.format(action_index))
        os.makedirs(intermediate_resutls_path, exist_ok=True)
        json_file = os.path.join(intermediate_resutls_path, '{}_action_data.json'.format(action_index))
        with open(json_file, 'r') as f:
            self.action_datas = json.load(f)



    def mk_cluster_dirs(self, save_dir, num_cls):
        '''
        save_dir : 保存分类结果的根目录
        num_cls : 分类的数量，种类数
        '''
        for i in range(num_cls):
            sub_dir = os.path.join(save_dir, str(i))
            if os.path.exists(sub_dir):
                shutil.rmtree(sub_dir)
            os.makedirs(sub_dir, exist_ok=True)

    def generate_main_imgs(self):
        '''在追踪结果的基础之上，生成各个动作的主图片。'''
        if os.path.exists(self.main_imgs_dir):
            shutil.rmtree(self.main_imgs_dir)
        os.makedirs(self.main_imgs_dir)

        FMLoader = self.FMLoader_dir
        if os.path.exists(FMLoader):
            print('{} exists'.format(FMLoader))
            action_indexes = os.listdir(FMLoader)
            action_indexes = sorted(action_indexes, key=lambda x: int(x))
            for action_index in action_indexes:
                action_dir = os.path.join(FMLoader, '{}'.format(action_index))
                if os.path.exists(action_dir):
                    target_read_path = os.path.join(action_dir, '0.jpg')
                    target_save_path = os.path.join(self.main_imgs_dir, '{}.jpg'.format(action_index))
                    shutil.copy(target_read_path, target_save_path)
        self.logger.log(24, 'SVHN_Predict.generate_main_imgs() Finished')

    def cluster_main_imgs(self):
        '''
        	:param ReID: ReID model
        	:param ReIDCfg: ReID configure
        	:param main_img_dir: The dir save the imgs which the programme what to cluster.
        	:param action_datas:
        	:param save_dir:
        	:param num_cls: how many classes that the programme want !
        	:return:
        	'''
        # 计时器
        cluster_main_imgs_timer = Timer()
        cluster_main_imgs_timer.tic()

        '''在追踪结果的基础之上，生成各个动作的主图片。'''
        self.generate_main_imgs()

        # 创建ReID模型
        self.ReID = ReID_Model(self.ReIDCfg)
        self.ReID.cuda()

        # make directories to save the clustered imgs.
        action_datas = self.action_datas

        # 场上有四类目标人物，创建四个子文件夹
        save_dir = self.main_imgs_dir
        self.mk_cluster_dirs(save_dir, self.num_cls)

        '''Preprocess the imgs before ReID'''
        if not os.path.exists(self.main_imgs_dir):
            raise ValueError("The main_img_dir is not exits")

        '''对要输入ReID网络的图片进行预处理'''
        imgs_arrays_all, img_names_all = ReID_imgs_load_by_home_and_away(self.ReIDCfg, self.main_imgs_dir, self.action_datas)

        # 分成主客两队
        cls_res_all = {'Home': 0, 'Away': 2} # 主队保存在前两个文件夹 0 和 1， 客队保存在后两个文件夹 2 和 3
        for TeanIndex, TeamType in enumerate(['Home', 'Away']):

            imgs_arrays = imgs_arrays_all[TeamType]
            img_names = img_names_all[TeamType]
            cls_res = cls_res_all[TeamType]
            all_feats = [] # 用来存储各个动作主图片的ReID特征
            with torch.no_grad():

                for imgs_array in imgs_arrays:
                    imgs_array = imgs_array.to('cuda')
                    feats = self.ReID(imgs_array).cpu().numpy().tolist()
                    all_feats.extend(feats)

            length = len(all_feats)
            self.logger.log(24, ' ReID models ,there are {} actions of TeamType {} want to be delt with.'.format(length,TeamType))

            '''根据ReID特征，进行分类，分成num_cls类, 门将和球员'''
            assignments, dataset = k_means(all_feats, 2)

            '''根据分类结果，将图片按文件夹分类'''
            for index, cls in enumerate(assignments):
                cls += cls_res # 所要保存的文件夹的序号
                # 是否有识别成功以号码检测为准。
                if int(action_datas[int(img_names[index])]['num']) == -1 or \
                        action_datas[int(img_names[index])]['num'] == None:

                    shutil.copyfile(os.path.join(self.main_imgs_dir, img_names[index] + '.jpg'),
                                    os.path.join(save_dir,'{}'.format(cls),'{}_.jpg'.format(img_names[index])))
                else:
                    shutil.copyfile(os.path.join(self.main_imgs_dir, img_names[index] + '.jpg'),
                                    os.path.join(save_dir,'{}'.format(cls),'{}_{}.jpg'.format(img_names[index], action_datas[int(img_names[index])]['num'])))

                action_datas[int(img_names[index])]['team'] = str(cls)

        self.action_datas = action_datas
        self.logger.log(24, 'SVHN_Predict.cluster_main_imgs() Finished， consums {}s'.format(cluster_main_imgs_timer.toc()))

