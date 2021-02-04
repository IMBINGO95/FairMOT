import shutil
import os


def makedir_v1(dir):
    # 先删除，后创建
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)