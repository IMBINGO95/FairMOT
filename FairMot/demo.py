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
logger = Log(__name__).getlog()

import FairMot.lib.datasets.dataset.jde as datasets
from FairMot.track import eval_seq


import cv2
def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    dataloader.cap.set(cv2.CAP_PROP_POS_MSEC, round(1000 * 120))
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename, save_dir=frame_dir, show_image=False, frame_rate=frame_rate)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
