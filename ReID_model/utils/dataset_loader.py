# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import os
import os.path as osp
import torch
from PIL import Image
from torch.utils.data import Dataset
from ReID_model.modeling import build_model
from ReID_model.transforms.build import build_transforms



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    try:
        img = Image.open(img_path).convert('RGB')
        got_img = True
        return img, got_img

    except IOError:
        print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
        
        return None, got_img


def model_load(ReIDCfg):
    model = build_model(ReIDCfg, 751)
    model.load_param(ReIDCfg.TEST.WEIGHT)
    return model


def img_preprocess(img_path, transform=None):
    img, got_img = read_image(img_path)
    if got_img == True:
        if transform is not None:
            img = transform(img)
        return img ,got_img
    else:
        return img, got_img

def ReID_imgs_load_by_home_and_away(ReIDCfg, img_dir, action_data):
    '''
    首先按队伍对球员进行一次区分
    接着按门将和队员，使用ReID来分类。
    '''
    val_transforms = build_transforms(ReIDCfg, is_train=False)
    # 筛选排序
    imgs_name = [ img_name for img_name in os.listdir(img_dir) if img_name.split('.')[-1] == 'jpg']
    imgs_name = sorted(imgs_name, key=lambda x : int(x.split('.')[0]))

    # 按主队和客队划分，保存中间参数
    imgs_arrays = {'Home': [], 'Away': []}
    count = {'Home': -1, 'Away': -1 }
    team_index = {'Home': 0, 'Away': 0 }
    names = {'Home': [], 'Away': []}

    for img_name in imgs_name:

        action_index = int(img_name.split('.')[0]) # 该图片在动作序列中对应的顺序
        TeamType = action_data[action_index]['teamType'] # 所属队伍

        # 有些文件中的 主客队是由 0 和 1 来划分的
        if type(TeamType) == int:
            if TeamType == 0:
                TeamType = 'Home'
            elif TeamType == 1:
                TeamType = 'Away'

        # 输入ReID网络的图片是一次不能太多， 因此需要每一百张划分一次
        if team_index[TeamType] % 100 == 0:
            count[TeamType] += 1
            imgs_arrays[TeamType].append([])

        # need to process damaged img, do something
        img, got_img = img_preprocess(os.path.join(img_dir, img_name), val_transforms)
        if got_img == False:
            continue

        imgs_arrays[TeamType][count[TeamType]].append(img.unsqueeze(dim=0))
        names[TeamType].append(img_name.split('.')[0])
        # 加入成功，总数 + 1
        team_index[TeamType] += 1

    # 把每个部分的imgs_array 拼起来成一个 Tensor
    for TeamType in imgs_arrays.keys():
        for array_index, array in enumerate(imgs_arrays[TeamType]):
            imgs_arrays[TeamType][array_index] = torch.cat(array, 0)

    return imgs_arrays, names


def ReID_imgs_loadv1(ReIDCfg, main_dir, index):

    val_transforms = build_transforms(ReIDCfg, is_train=False)
    imgs_arrays = []
    
    # load in main imgs
    main_sub_img, got_img = img_preprocess(os.path.join(main_dir,'main_imgs','{}.jpg'.format(index))
                                  ,val_transforms)
    if got_img == False:
        return imgs_arrays, got_img
    
    imgs_arrays.append(main_sub_img.unsqueeze(dim=0))

    img_dir = os.path.join(main_dir,'visualization','{}'.format(index),'NumReg')
    imgs_name = sorted(os.listdir(img_dir),key=lambda x: int(x.split('_')[0]))
    for index, img_name in enumerate(imgs_name):
        if img_name.split('.')[-1] != 'jpg':
            continue
        # need to process damaged img, do something
        # print(index,img_name)
        img,got_img = img_preprocess(os.path.join(img_dir, img_name), val_transforms)
        if got_img == False:
            continue
        imgs_arrays.append(img.unsqueeze(dim=0))
        
    imgs_arrays = torch.cat(tuple(imgs_arrays), 0)
    return imgs_arrays, True

def transform_imgs(ReIDCfg,imgs):
    
    val_transforms = build_transforms(ReIDCfg, is_train=False)
    
    imgs_arrays = []
    for index, img in enumerate(imgs):
        # need to process damaged img, do something
        if img.size == 0:
            continue
        img = Image.fromarray(img,'RGB')
        img = val_transforms(img)
        imgs_arrays.append(img.unsqueeze(dim=0))

    imgs_arrays = torch.cat(tuple(imgs_arrays), 0)
    return imgs_arrays

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path
