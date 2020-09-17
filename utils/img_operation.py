import cv2
import os
from CalibrateTransfer.data_preprocess import write_data_to_json_file,read_data_from_json_file,make_dir,read_subdata,read_stack_data
from utils.log import Log
import logging
logger = Log('test_log', __name__).getlog()


class Save_sub_imgs():
    def __init__(self,opt, C_T_output_queue,queueSize=1024):

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

        self.output_Q = C_T_output_queue

    def save_img(self):

        self.virsual_dir = os.path.join(self.root_path,'vis')
        os.makedirs(self.virsual_dir,exist_ok=True)

        for i in range(len(self.action_datas)):
            Flag ,(action_index , sub_imgs, ReID_features) = self.output_Q.get()

            if Flag == False:

                continue

            vis_dir = os.path.join(self.virsual_dir,'{}'.format(i))
            os.makedirs(vis_dir,exist_ok=True)
            for img_index in  range(len(sub_imgs)):
                sub_img = sub_imgs[img_index]
                cv2.imwrite(os.path.join(vis_dir,'{}.jpg'.format(img_index)),sub_img)

            logger.info('imgs saved in vis_dir ==== action {}'.format(i))


