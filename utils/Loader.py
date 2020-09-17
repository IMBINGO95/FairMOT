# import os
# import sys
# from threading import Thread
# from queue import Queue
#
# import cv2
# import scipy.misc
# import numpy as np
#
# from CalibrateTransfer.img_operation import ScreenSHot
# from CalibrateTransfer.data_preprocess import write_data_to_json_file,read_data_from_json_file,make_dir,read_subdata,read_stack_data
# from CalibrateTransfer.cv_transfer import transform_2d_to_3d,object_To_pixel,updata_img_point
# from CalibrateTransfer.img_operation import GenerateRect
#
# import torch
# import torch.multiprocessing as mp
#
# from FairMot.track import Short_track_eval
#
# class LoadShortCutVideo:  # for short tracking
#     def __init__(self,video, video_time, rect, Output_size, img_size=(1088, 608), multiple = 2):
#
#         self.cap = video
#         self.cap.set(cv2.CAP_PROP_POS_MSEC,round(1000*video_time))  # 将视频设置到动作发生的时间
#         self.current_frame_index = self.cap.get(cv2.CAP_PROP_POS_FRAMES) # 动作发生的时间对应的帧
#         self.multiple = multiple # 获取 2 * multiple倍 的视频帧率长度的图片
#         self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS))) # 计算帧率
#         self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index - multiple*self.frame_rate) # 将视频向前调整 multiple 秒
#
#         self.width, self.height = img_size[0] , img_size[1] # 网络输入的Feature Map的大小
#         [self.vw, self.vh] = Output_size # 输入图片的大小
#         [self.w, self.h] = Output_size # 可视化的图片的大小
#
#         self.rect = rect # 对应的目标区域 [x_l,y_l,x_r,y_r]
#         self.count = 0
#         self.vn = 2 *multiple * self.frame_rate + 1
#         print('Lenth of the video: {:d} frames'.format(self.vn))
#
#     def get_size(self, vw, vh, dw, dh):
#         wa, ha = float(dw) / vw, float(dh) / vh
#         a = min(wa, ha)
#         return int(vw * a), int(vh * a)
#
#     def __iter__(self):
#         self.count = -1
#         return self
#
#     def __next__(self):
#         # Read image
#         res, img0 = self.cap.read()  # BGR
#         assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
#         # 裁剪图片
#         img0 = img0[self.rect[1]:self.rect[3],self.rect[0]:self.rect[2]]
#         # Normalize RGB
#         img = img0[:, :, ::-1].transpose(2, 0, 1)
#         img = np.ascontiguousarray(img, dtype=np.float32)
#         img /= 255.0
#         # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
#         self.count += 1
#         if self.count == len(self):
#             raise StopIteration
#
#         return self.count, img, img0
#
#     def __len__(self):
#         return self.vn  # number of files
#
# class ImgSequenceLoader:
#     def __init__(self, opt, dataloder, queueSize=1000, sp=False):
#
#         self.dir_name = opt.dir_name
#         self.root_path = os.path.join(opt.data_root, '{}'.format(opt.dir_name))
#         # logger.info('目标文件夹是{}'.format(root_path))
#
#         self.file_name = opt.file_name
#         # 本来就是要载入两次视频，分开读亦可以
#         self.Videoparameters, \
#         self.setting_parameter, \
#         self.action_datas, \
#         self.channel_list, \
#         self.parameter = read_data_from_json_file(self.root_path, self.file_name, opt)
#
#         self.stopped = False
#         self.datalen = len(self.action_datas)
#
#
#         # initialize the queue used to store frames read from
#         # the video file
#         self.sp = sp
#         if sp:
#             self.Q = Queue(maxsize=queueSize)
#         else:
#             self.Q = mp.Queue(maxsize=queueSize)
#
#     def start(self):
#         # start a thread to read frames from the file video stream
#         if self.sp:
#             t = Thread(target=self.update, args=())
#             t.daemon = True
#             t.start()
#         else:
#             p = mp.Process(target=self.update, args=())
#             p.daemon = True
#             p.start()
#         return self
#
#     def update(self):
#         # keep looping the whole dataset
#
#
#         for index in range(self.datalen):
#
#             # result_root = make_dir(self.root_path, index, Secondary_directory='{}_short_tracking'.format(self.dir_name))
#
#             '''read each item from  subdata of action datas according to the index '''
#             channel, action_time, img_point, video_parameter = read_subdata(self.action_datas[index], self.Videoparameters)
#
#             video = video_parameter['video']
#             # action time need to add the delta time to calibrate the time between channels .
#             video_time = action_time + video_parameter['delta_t']
#             width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
#             height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
#             Message = GenerateRect(img_point, self.setting_parameter['Output_size'], self.setting_parameter['bias'], width,
#                                    height)
#
#             if Message[0] == True:
#                 # 获取目标区域
#                 rect = Message[1]
#                 x_l = int(rect[0])
#                 y_l = int(rect[1])
#                 x_r = int(rect[2] + rect[0])
#                 y_r = int(rect[3] + rect[1])
#                 rect = [x_l, y_l, x_r, y_r]
#                 # 目标点坐标相对于从原图中的位置，更新到相对于截图中的位置
#                 reference_point = (int(img_point[0] - x_l), int(img_point[1] - y_l))
#                 # sub_img = img[y_l:y_r, x_l:x_r]
#             else:
#                 # 如果没有截图则,则无需放入Queue中。
#                 self.Q.put(None,None,None)
#                 continue
#
#             # logger.info('Starting tracking...')
#             dataloader = LoadShortCutVideo(video, video_time, rect, self.setting_parameter['Output_size'])
#             target_frame = dataloader.multiple * dataloader.frame_rate
#
#             # result_filename = os.path.join(result_root, '..', '{}.txt'.format(index))
#             frame_rate = dataloader.frame_rate
#
#             img, orig_img, im_name, im_dim_list = self.dataloder.getitem(
#                 self.Q.put((orig_img[k], im_name[k], boxes_k, scores[dets[: ,0 ]= =k], inps, pt1, pt2))
#
#     def read(self):
#         # return next frame in the queue
#         return self.Q.get()
#
#     def len(self):
#         # return queue len
#         return self.Q.qsize()