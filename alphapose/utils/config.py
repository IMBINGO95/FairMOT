import yaml
from easydict import EasyDict as edict
import os

def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config

# 从yaml中读取配置
def readyaml(file):
    if os.path.isfile(file):
        fr = open(file, 'r')
        config = yaml.load(fr)
        fr.close()
        return config
    return None
# 向yaml文件写入配置
def writeyaml(file,data):
    fr = open(file,'w')
    yaml.dump(data,fr)
    fr.close()

# 将EasyDict 写成 普通得dict的形式。
def TransferEasyDictToDICT(TargetEdict):

    output_dict = {}
    for key in TargetEdict.keys():
        if type(TargetEdict[key]) == type(edict()):
            output_dict[key] = TransferEasyDictToDICT(TargetEdict[key])
        else:
            output_dict[key] = TargetEdict[key]
    return output_dict

if __name__ == '__main__' :
   
    yaml_file = "/datanew/hwb/FairMOT-master/alphapose/configs/coco/resnet/256x192_res50_lr1e-3_1x-simple.yaml"
   
    config1 = update_config(yaml_file)
    config2 = readyaml(yaml_file)

    yaml_write1 = 'y1.yaml'
    yaml_write2 = 'y2.yaml'



    writeyaml(yaml_write1, config1)
    writeyaml(yaml_write2, config2)

    config1_ = readyaml(yaml_write1)
    config2_ = readyaml(yaml_write2)
    keys = config1_.keys()
    output_item = {}
    valus = config1_.values()

    print(type(config1))
    config1_2 = TransferEasyDictToDICT(config1)

    print(config1)
    print(config2)
