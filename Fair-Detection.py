from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths

import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import os.path as osp
from FairMot.lib.opts import opts
from FairMot.lib.tracking_utils.utils import mkdir_if_missing
from utils.log import Log
logger = Log(__name__,__name__).getlog()
logger.setLevel(logging.INFO)

import FairMot.lib.datasets.dataset.jde as datasets
from FairMot.lib.tracker.multitracker import JDETracker
from FairMot.track import detect

import re
import cv2
def regular_videoName(root_path):
	'''
	given an string regular rule, sorted the video names by this regular rule
	:param root_path: the path includes the videos
	:return: the target videoname dictionary.
	'''
	# bulid the regular format
	re_videoName = re.compile(r'(c|C)h0\w*.mp4')
	videoNames = {}
	for videoName in os.listdir(root_path):
		if re_videoName.match(videoName):
			videoNames[videoName[3]] = videoName
	return  videoNames

def demo(opt):

    tracker = JDETracker(opt, frame_rate=30) # What is JDE Tracker?

    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting detection...')
    root_dir = '/datanew/hwb/data/Football/SoftWare/0'
    channels = regular_videoName(root_dir)
    for channel in channels.keys():
        video_name = channels[channel]
        print('Starting to detect {}/{}'.format(root_dir,video_name))
        input_video = os.path.join(root_dir,video_name)
        dataloader = datasets.LoadVideo(input_video, opt.img_size,gap=1000)
        dataloader.cap.set(cv2.CAP_PROP_POS_MSEC, round(1000 * 120))
        result_filename = os.path.join(result_root, 'results.txt')
        # frame_rate = dataloader.frame_rate
        frame_dir = os.path.join(root_dir, 'detection',video_name[0:4])
        os.makedirs(frame_dir, exist_ok=True)
        # detect(opt,tracker,dataloader, 'mot', result_filename, save_dir=frame_dir, show_image=False)
        detect(opt, tracker, dataloader, dir_id=1, save_dir=frame_dir, show_image=True)

    # if opt.output_format == 'video':
    #     output_video_path = osp.join(result_root, 'result.mp4')
    #     cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
    #     os.system(cmd_str)

if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
