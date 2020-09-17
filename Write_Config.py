# 追踪器的参数
from FairMot.lib.opts import opts
from config.SVHN.CC import Config
from opt import OPT_setting
from config.ReID import cfg as ReIDCfg
from easydict import EasyDict as edict
from alphapose.utils.config import update_config

import os
import yaml

def set_tracking_opt():
    Tracker_opt = opts().init()
    return Tracker_opt

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

if __name__ == "__main__":

    opt = OPT_setting().init()
    #############################
    # Load in Trackor  parameter#
    #############################
    Tracker_opt = set_tracking_opt()
    Tracker_opt_ = edict(vars(Tracker_opt))
    Tracker_opt__ = TransferEasyDictToDICT(Tracker_opt_)
    writeyaml('./config/FairMot/defaults.yaml',Tracker_opt__)

    ########################################
    # Load in ReID Classification Parameter#
    ########################################
    print('===> Start to constructing and loading ReID model', ['yellow', 'bold'])
    if opt.ReIDCfg != "":
        ReIDCfg.merge_from_file(opt.ReIDCfg)
    ReIDCfg.freeze()

    ReIDCfg_ = edict(ReIDCfg)
    ReIDCfg__ = TransferEasyDictToDICT(ReIDCfg_)
    writeyaml('./config/ReID/defaults.yaml', ReIDCfg__)

    ##########################
    # Load in Poser Parameter#
    ##########################
    Pose_opt = update_config(opt.Poser_cfg)

    Pose_opt_ = edict(Pose_opt)
    Pose_opt__ = TransferEasyDictToDICT(Pose_opt_)
    writeyaml('./config/alphapose/defaults.yaml', Pose_opt__)

    ##################################
    # Load in Number Predictor Number#
    ##################################
    Num_Pred_opt = Config.fromfile(opt.SvhnCfg)
    Num_Pred_opt_ = edict(Num_Pred_opt)
    Num_Pred_opt__ = TransferEasyDictToDICT(Num_Pred_opt_)
    writeyaml('./config/SVHN/defaults.yaml', Num_Pred_opt__)
