import os

def get_index(dir_path):
    '''获取这个文件架中所有已经计算过了的index'''

    dir_indexs = [dir_index for dir_index in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path,dir_index))]
    if len(dir_indexs) == 0:
        return [0]
    else:
        # 按index大小从小到大排序
        dir_indexs = map(int,dir_indexs) # 转换类型，
        dir_indexs = sorted(dir_indexs) #  从小到大排序
        return dir_indexs