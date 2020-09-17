import yaml
from easydict import EasyDict as edict

# 将所有模型的配置文件读取进来
def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config
