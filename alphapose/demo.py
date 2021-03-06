"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import json
import sys
import time
import cv2
import numpy as np
import torch
from tqdm import tqdm

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectedImgsLoader
from alphapose.utils.pPose_nms import write_json
from alphapose.utils.transforms import flip, flip_heatmap, heatmap_to_coord_simple
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.writer import DataWriter, heat_to_map
"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str,
                    default='/datanew/hwb/FairMOT-master/alphapose/configs/coco/resnet/256x192_res50_lr1e-3_1x-simple.yaml',
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str,
                    default='/datanew/hwb/AlphaPose-master/pretrained_models/simple_res50_256x192.pth',
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',default='/datanew/hwb/data/Football/SoftWare/{}/{}_detection/subimg',
                    help='image-directory')
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=True, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="/datanew/hwb/data/Football/SoftWare/8/ch02_20190804170807_20200113185343.mp4")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=True, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video', action='store_true', default=False)

args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = (args.detector == 'tracker')

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

def print_finish_info():
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


if __name__ == "__main__":

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print(f'Loading pose model from {args.checkpoint}...')
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()

    for game in ['FM']:
        # Load detected imgs

        input_source = '/datanew/hwb/data/WG_Num/{}/JPEGImages'.format(game)
        print(input_source)

        outputpath = os.path.join(input_source,'..','{}_vis'.format(game))
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)

        json_file = os.path.join(input_source,'..','{}_vis_keypoints.json'.format(game))

        data = []
        det_loader = DetectedImgsLoader(input_source, cfg, args, batchSize=args.detbatch, queueSize=args.qsize)
        det_worker = det_loader.start()

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }
        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

        batchSize = args.posebatch
        if args.flip:
            batchSize = int(batchSize / 2)
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inp, orig_img, cropped_boxes, img_name) = det_loader.read()
                if orig_img is None:
                    break
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                inps = torch.unsqueeze(inp,dim=0)
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], det_loader.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 21
                    hm.append(hm_j)
                hm = torch.cat(hm)

                vis_img, keypoints = heat_to_map(orig_img,hm,cropped_boxes)

                cv2.imwrite(os.path.join(outputpath,img_name),vis_img)
                data.append({'img_name':img_name,'keypoints':keypoints})

                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                # hm_ = hm_.cpu()
                # print('cropped_boxes',cropped_boxes)
                # writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, os.path.basename(im_name))

                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
        print_finish_info()

        with open(json_file,'w') as f :
            json.dump(data,f)

        det_loader.stop()