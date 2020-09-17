
import os
import collections
import json
import cv2
import time

from utils_BINGO import divide_img_range
from utils_BINGO import gtsToxml

def mkdataset_in_VOC_format(im_path,save_dir,gt_json_path ,dataset_name,section_num, mode=0):
    '''
    To make a detection dataset in VOC format.
    :param im_path: Original img path that save the imgs
    :param save_dir:  The target path that we save the transformed imgs
    :param gt_json_path:  gts in json format, gt in this [which frame,ID,x,y,w,h,channel,jersey_number]
    :param dataset_name:
    :param section_num:  [x,y] means we want to divide the original img to x*y parts
    :param mode:  mode = 0 means to create a traning dataset, mode = 1 means to create a testing dataset.
    :return:  None
    '''

    dataset_path = os.path.join(save_dir, dataset_name)
    img_save_path = os.path.join(dataset_path,'JPEGImages/')
    xml_save_path = os.path.join(dataset_path, 'Annotations/')
    txt_save_path = os.path.join(dataset_path,'ImageSets/Main/')

# mk folders in VOC format!
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    if not os.path.exists(xml_save_path):
        os.makedirs(xml_save_path)

    if not os.path.exists(txt_save_path):
        os.makedirs(txt_save_path)
# to creat a .txt file to save the img ID that we want to train on or test on !
    if mode == 0:
        txt_save_path = txt_save_path + 'train.txt'
    else:
        txt_save_path = txt_save_path + 'test.txt'
    txt_file = open(txt_save_path, 'w')

    # create two OrderDict to contains gts.
    # seperate the gts by frame first
    # seperate the gts in one frame to each parts
    gts_by_frame = collections.OrderedDict()
    gts_seperate = collections.OrderedDict()
    # diliver each gt to their own dict
    with open(gt_json_path, 'r') as f:
        gts = json.load(f)
    for gt in gts:
        frame = '{:0>6}'.format(gt[0])
        # change [which frame,ID,xl,yl,w,h,c,NUM] to [which frame,ID,xl,yl,xr,yr,c,NUM]
        gt[4] = gt[2] + gt[4]
        gt[5] = gt[3] + gt[5]
        if frame not in gts_by_frame:
            gts_by_frame[frame] = []
            gts_seperate[frame] = collections.OrderedDict()
            gts_by_frame[frame].append(gt)
        else:
            gts_by_frame[frame].append(gt)

    # read all the img_name in the origin_img folder.
    im_fnames = sorted(
        (os.path.splitext(fname)[0] for fname in os.listdir(im_path) if os.path.splitext(fname)[-1] == '.jpg'))


    for i, frame in enumerate(im_fnames):

        image = cv2.imread(im_path + frame + '.jpg', cv2.IMREAD_COLOR)
        loop_start = time.time()

        im_width, im_height = image.shape[1], image.shape[0]

        # deviede img into several part!
        section_w, box_range_w, part_w = divide_img_range(im_width, im_height, section_num[0])
        section_h, box_range_h, part_h = divide_img_range(im_width, im_height, section_num[1], mode=1)

        # Traverse by width
        for part_w in range(len(section_w)):
            _, _, xl, xr = section_w[part_w]
            # Traverse by height
            for part_h in range(len(section_h)):
                yl, yr, _, _ = section_h[part_h]
                part_img = image[yl:yr, xl:xr]
                part_num = '_{:0>2}_{:0>2}'.format(part_w + 1, part_h + 1)
                gts_seperate[frame][part_num] = []
                # deliver every gt to different parts,
                # because different parts have overlapping places
                # so a gt can be diliver to more than one parts
                for gt in gts_by_frame[frame]:
                    if gt[2] > xl and gt[3] > yl and gt[4] < xr and gt[5] < yr:
                        xml = []
                        # the label of this gt ,must be str
                        xml.append('person')
                        # the pose of this gt ,must be str
                        xml.append('stand')
                        # Did this gt truncate? 0 for no, 1 for yes. ,must be str
                        xml.append('0')
                        # Is this gt difficult to detect? 0 for no, 1 for yes. ,must be str
                        xml.append('0')
                        # adjust [xl,yl,xr,yr] from relative to (0,0) of the original img to part img .
                        xml.append(gt[2] - xl)  # xl
                        xml.append(gt[3] - yl)  # yl
                        xml.append(gt[4] - xl)  # xr
                        xml.append(gt[5] - yl)  # yr
                        gts_seperate[frame][part_num].append(xml)
                # only save the imgs that include gts.
                if len(gts_seperate[frame][part_num]) > 0:
                    # draw rectangle on the imgs.
                    # part_img = draw_detection(part_img, gts_seperate[frame][part_num])
                    file_name = img_save_path + frame + part_num + '.jpg'
                    cv2.imwrite(file_name, part_img)
                    # write gts into xml file .
                    gtsToxml(dataset_name=dataset_name, file_dir=xml_save_path, img_ID=frame + part_num, \
                             size=part_img.shape, gts=gts_seperate[frame][part_num])

                    txt_file.writelines(frame + part_num + '\n')
        print('frame {:0>6}'.format(frame) + ' have been processed!  ')
    txt_file.close()