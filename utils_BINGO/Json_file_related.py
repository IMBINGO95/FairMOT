import matplotlib.pyplot as plt
import json
import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm
from utils_BINGO.xml_related import *

def expand_bbox(left, right, top, bottom, img_width, img_height,ratio = 0.1, expand_w_min = 10):
    '''
    以一定的ratio向左右外扩。 不向上向下扩展了。
    '''
    width = right - left
    height = bottom - top
     # expand ratio
    expand_w_min = max(ratio * width , expand_w_min) # 最小外扩 expand_w_min
    new_left = np.clip(left - expand_w_min, 0, img_width)
    new_right = np.clip(right + expand_w_min, 0, img_width)
    # new_top = np.clip(top - ratio * height, 0, img_height)
    # new_bottom = np.clip(bottom + ratio * height, 0, img_height)

    return [int(new_left), int(new_right), int(top), int(bottom)]

def get_back_box(keypoints, img_height, img_width, ratio=0.1, expand_w_min=10):
    '''这个get box 是用来获取球员的背部区域的'''
    xmin = min(keypoints[5 * 3], keypoints[11 * 3])
    xmax = max(keypoints[6 * 3], keypoints[12 * 3])
    ymin = min(keypoints[5 * 3 + 1], keypoints[6 * 3 + 1])
    ymax = max(keypoints[11 * 3 + 1], keypoints[12 * 3 + 1])

    return [int(round(xmin)), int(round(xmax))], expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height, ratio ,expand_w_min)

def get_front_box(keypoints, img_height, img_width, ratio=0.1, expand_w_min=10):
    '''这个get box 是用来获取球员的正面胸部区域的'''
    xmax = max(keypoints[5 * 3], keypoints[11 * 3])
    xmin = min(keypoints[6 * 3], keypoints[12 * 3])
    ymin = min(keypoints[5 * 3 + 1], keypoints[6 * 3 + 1])
    ymax = max(keypoints[11 * 3 + 1], keypoints[12 * 3 + 1])

    return [int(round(xmin)), int(round(xmax))], expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height, ratio ,expand_w_min)
def make_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def filter_outliers(img_dir,dir_save_front, dir_save_True, dir_save_False,json_file,save_rectangle=True,mode='test'):
    # 基于骨骼关键节点的信息，通过肩宽和半身长，来筛选符合条件的目标。
    print(json_file)
    with open(json_file,'r') as f :
        data = json.load(f)

    data_len = len(data)

    count_right_pose = 0
    count_final = 0
    count_True = 0
    count_False = 0
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
    all_aspect_ratios = []
    if mode in ['test','train']:
        anno_dir_read = os.path.join(img_dir,'..','Annotations')
        anno_dir_save = os.path.join(img_dir,'..','Annotations_save')
        make_dir(anno_dir_save)

        target_transform = AnnotationTransform(['region'])

    for i in im_names_desc:
        item = data[i]
        img_name = item['img_name']
        id = img_name.split('.')[0]
        img = cv2.imread(os.path.join(img_dir,img_name))
        height,width,_ = img.shape
        keypoints = item['keypoints']

        # 这个判断标准和get_box的标准不一样。
        # 用来判断是否背向的
        l_x_max = max(keypoints[5 * 3], keypoints[11 * 3])
        r_x_min = min(keypoints[6 * 3], keypoints[12 * 3])
        t_y_max = max(keypoints[5 * 3 + 1], keypoints[6 * 3 + 1])
        b_y_min = min(keypoints[11 * 3 + 1], keypoints[12 * 3 + 1])

        if l_x_max < r_x_min and t_y_max < b_y_min:
            '初步判断球员是否背向'
            [xmin_old, xmax_old], [xmin, xmax, ymin, ymax] = get_back_box(keypoints, height, width, ratio=0.1, expand_w_min=10)

            count_right_pose += 1
            if height < 130 or width < 60:
                continue

            count_final += 1
            #计算肩宽、胯宽和半身长
            Shoulder_width = keypoints[6*3] - keypoints[5*3]
            Crotch_width = keypoints[12*3] - keypoints[11*3]
            body_length = ymax - ymin
            if body_length == 0 :
                print(os.path.join(img_dir,img_name))
            aspect_ratio = (max(Shoulder_width,Crotch_width)) / (body_length)
            all_aspect_ratios.append(aspect_ratio)

            if aspect_ratio >= 0.40:
                dir_save = dir_save_True
                count_True += 1
                # 保存Annotations
                if mode in ['test','train']:
                    xml_read_path = os.path.join(anno_dir_read,'{}.xml'.format(id))
                    width_read, height_read, depth_read, length, number = read_xml(xml_read_path, target_transform)
                    if width != int(width_read) or height != int(height_read):
                        raise ValueError("{} is not right".format(type(id)))
                    else:
                        # xml_write_path = os.path.join(anno_dir_save, '{}.xml'.format(id))
                        write_xml(anno_dir_save,width_read,height_read,depth_read,id,length,num=number,
                                  item=[max(xmin,0), max(0,ymin), min(xmax,width), min(ymax,height)])

            else:
                dir_save = dir_save_False
                count_False += 1

            if save_rectangle == True:
                img_rectangle = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(os.path.join(dir_save, img_name), img_rectangle)
            else:
                # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1)
                cv2.rectangle(img, (xmin_old, ymin), (xmax_old, ymax), color=(255, 0, 0), thickness=1)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)
                cv2.imwrite(os.path.join(dir_save, img_name), img)



    print('count_right_pose / length = {} / {} = {}'.format(count_right_pose, data_len, count_right_pose / data_len))
    print('count_final / length = {} / {} = {}'.format(count_final, data_len, count_final / data_len))
    print('count_True / length = {} / {} = {}'.format(count_True, data_len, count_True / data_len))
    print('count_False / length = {} / {} = {}'.format(count_False, data_len, count_False / data_len))

    his = np.array(all_aspect_ratios)
    scale = np.histogram(his, bins=100, range=(0, 1))
    num = 'num:{:->8}\n'.format(len(his))
    max_score = 'max:{:.4f},'.format(np.max(his))
    min_score = 'min:{:.4f}\n'.format(np.min(his))
    mean_score = '(r)mean:{:.4f},'.format(np.mean(his))
    median_score = '(g)median:{:.4f}'.format(np.median(his))

    plt.hist(his, bins=100, range=(0, 1))
    '''draw mean and median line in the scores histogram'''
    plt.axvline(x=np.mean(his), ymin=np.min(scale[0]), ymax=np.max(scale[0]), linewidth=5, color='r')
    plt.axvline(x=np.median(his), ymin=np.min(scale[0]), ymax=np.max(scale[0]), linewidth=5, color='g')

    plt.title(mode)
    plt.ylabel('count')
    plt.xlabel(num + max_score + min_score + mean_score + median_score)

    plt.grid(True)
    plt.subplots_adjust(hspace=0.5)  # set gap between subplot !
    plt.tight_layout()
    # plt.savefig(os.path.join(dir, title + '_' + file + '.png'))
    plt.show()
    plt.close()

def filter_outliers_Negative(img_dir,json_file,save_rectangle=True,mode='test',vis=False):
    # 基于骨骼关键节点的信息，通过肩宽和半身长，来筛选符合条件的目标。
    # 这次筛选的是 正面的球员
    print(json_file)
    with open(json_file,'r') as f :
        data = json.load(f)

    data_len = len(data)

    count_right_pose = 0
    count_final = 0
    count_True = 0
    count_False = 0
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
    all_aspect_ratios = []
    # if mode in ['Negative']:
    #     # anno_dir_read = os.path.join(img_dir,'..','Annotations')
    #     anno_dir_save = os.path.join(img_dir,'..','Annotations_save')
    #     make_dir(anno_dir_save)

        # target_transform = AnnotationTransform(['region'])
    mode = 'train'
    anno_dir_save = os.path.join(img_dir, '..', mode, 'Annotations')
    dir_save_True = os.path.join(img_dir, '..', mode, 'JPEGImages')
    dir_save_False = os.path.join(img_dir, '..', mode, 'False')
    os.makedirs(anno_dir_save,exist_ok=True)
    os.makedirs(dir_save_True,exist_ok=True)
    os.makedirs(dir_save_False,exist_ok=True)


    for i in im_names_desc:

        item = data[i]
        img_name = item['img_name']
        # if img_name == '17_2_A_Player_N_47_4.jpg.jpg':
        #     print()
        if len(img_name.split('.')) > 2:
            print(img_name)
            continue
        id = img_name.split('.')[0]
        img = cv2.imread(os.path.join(img_dir,img_name))
        if type(img)  != np.ndarray:
            print(img_name)
            continue
        height,width,depth = img.shape
        keypoints = item['keypoints']

        # 这个判断标准和get_box的标准不一样。
        # 用来判断是否背向的
        l_x_min = min(keypoints[5 * 3], keypoints[11 * 3]) # 左侧最小值
        r_x_max = max(keypoints[6 * 3], keypoints[12 * 3]) # 右侧最大值
        t_y_max = max(keypoints[5 * 3 + 1], keypoints[6 * 3 + 1])
        b_y_min = min(keypoints[11 * 3 + 1], keypoints[12 * 3 + 1])

        if l_x_min > r_x_max and t_y_max < b_y_min:
            '初步判断球员是否正向'
            [xmin_old, xmax_old], [xmin, xmax, ymin, ymax] = get_front_box(keypoints, height, width, ratio=0.1, expand_w_min=10)

            count_right_pose += 1
            if height < 130 or width < 60:
                continue

            count_final += 1
            #计算肩宽、胯宽和半身长
            Shoulder_width = abs(keypoints[6*3] - keypoints[5*3])
            Crotch_width = abs(keypoints[12*3] - keypoints[11*3])
            body_length = ymax - ymin
            if body_length == 0 :
                print(os.path.join(img_dir,img_name))
            aspect_ratio = (max(Shoulder_width,Crotch_width)) / (body_length)
            all_aspect_ratios.append(aspect_ratio)

            if aspect_ratio >= 0.40:
                dir_save = dir_save_True
                count_True += 1
                # 保存Annotations
                length = 0
                number = -1

                write_xml(anno_dir_save,width,height,depth,id,length,num=number,
                          item=[max(xmin,0), max(0,ymin), min(xmax,width), min(ymax,height)])

            else:
                dir_save = dir_save_False
                count_False += 1

            if save_rectangle == True:
                img_rectangle = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(os.path.join(dir_save, '{}.jpg'.format(id)), img_rectangle)
            else:
                # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1)
                if vis == True:
                    cv2.rectangle(img, (xmin_old, ymin), (xmax_old, ymax), color=(255, 0, 0), thickness=1)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)
                cv2.imwrite(os.path.join(dir_save, img_name), img)

            if count_True == 3500:
                mode = 'test'
                anno_dir_save = os.path.join(img_dir, '..', mode, 'Annotations')
                dir_save_True = os.path.join(img_dir, '..', mode, 'JPEGImages')
                dir_save_False = os.path.join(img_dir, '..', mode, 'False')
                os.makedirs(anno_dir_save, exist_ok=True)
                os.makedirs(dir_save_True, exist_ok=True)
                os.makedirs(dir_save_False, exist_ok=True)

            elif count_True == 4500:
                break


    print('count_right_pose / length = {} / {} = {}'.format(count_right_pose, data_len, count_right_pose / data_len))
    print('count_final / length = {} / {} = {}'.format(count_final, data_len, count_final / data_len))
    print('count_True / length = {} / {} = {}'.format(count_True, data_len, count_True / data_len))
    print('count_False / length = {} / {} = {}'.format(count_False, data_len, count_False / data_len))

    his = np.array(all_aspect_ratios)
    scale = np.histogram(his, bins=100, range=(0, 1))
    num = 'num:{:->8}\n'.format(len(his))
    max_score = 'max:{:.4f},'.format(np.max(his))
    min_score = 'min:{:.4f}\n'.format(np.min(his))
    mean_score = '(r)mean:{:.4f},'.format(np.mean(his))
    median_score = '(g)median:{:.4f}'.format(np.median(his))

    plt.hist(his, bins=100, range=(0, 1))
    '''draw mean and median line in the scores histogram'''
    plt.axvline(x=np.mean(his), ymin=np.min(scale[0]), ymax=np.max(scale[0]), linewidth=5, color='r')
    plt.axvline(x=np.median(his), ymin=np.min(scale[0]), ymax=np.max(scale[0]), linewidth=5, color='g')

    plt.title(mode)
    plt.ylabel('count')
    plt.xlabel(num + max_score + min_score + mean_score + median_score)

    plt.grid(True)
    plt.subplots_adjust(hspace=0.5)  # set gap between subplot !
    plt.tight_layout()
    # plt.savefig(os.path.join(dir, title + '_' + file + '.png'))
    plt.show()
    plt.close()

def validate_false_pose(list_dir,pose_dir,origin_img_dir,save_dir):
    imgs = os.listdir(list_dir)
    data_len = len(imgs)
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    for i in im_names_desc :
        img = imgs[i]
        img_name = img.split('.')[0]
        shutil.copy(os.path.join(list_dir,img),os.path.join(save_dir,'{}_{}.jpg'.format(img_name,'一')))
        shutil.copy(os.path.join(pose_dir,img),os.path.join(save_dir,'{}_{}.jpg'.format(img_name,'二')))
        shutil.copy(os.path.join(origin_img_dir,img),os.path.join(save_dir,'{}_{}.jpg'.format(img_name,'三')))

def generate_positive_SVHN_annotation():
    for game in ['train','test']:

        dir = '/datanew/hwb/data/WG_Num/{}'.format(game)
        img_dir = '/datanew/hwb/data/WG_Num/{}/JPEGImages'.format(game)
        dir_save_front = '/datanew/hwb/data/WG_Num/{}/{}_front_after_sort'.format(game,game)
        dir_save_True = '/datanew/hwb/data/WG_Num/{}/{}_True_after_sort'.format(game,game)
        dir_save_False = '/datanew/hwb/data/WG_Num/{}/{}_False_after_sort'.format(game,game)
        make_dir(dir_save_True)
        make_dir(dir_save_False)
        file = '{}_vis_keypoints.json'.format(game)
        json_file = os.path.join(dir, file)
        filter_outliers(img_dir,dir_save_front, dir_save_True, dir_save_False, json_file, save_rectangle = True ,mode=game)

def generate_negative_SVHN_annotation():
    for game in ['Negative']:

        dir = '/datanew/hwb/data/WG_Num/{}'.format(game)
        img_dir = '/datanew/hwb/data/WG_Num/{}/JPEGImages'.format(game)

        file = '{}_vis_keypoints.json'.format(game)
        json_file = os.path.join(dir, file)
        filter_outliers_Negative(img_dir, json_file,save_rectangle = False ,mode=game)

if __name__ == '__main__':

    generate_negative_SVHN_annotation()
