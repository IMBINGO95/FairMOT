import os
import shutil

def generate_main_imgs(root, dir_indexes, move_file = False):

    for dir_index in dir_indexes:
        main_dir = os.path.join(root, '{}/intermediate_results/main_imgs'.format(dir_index))
        if os.path.exists(main_dir):
            shutil.rmtree(main_dir)
        os.makedirs(main_dir)

        FMLoader = os.path.join(root, '{}/intermediate_results/FMLoader'.format(dir_index))
        if os.path.exists(FMLoader):
            print('{} exists'.format(FMLoader) )
            action_indexes = os.listdir(FMLoader)
            action_indexes = sorted(action_indexes, key=lambda x:int(x))
            for action_index in action_indexes:
                action_dir = os.path.join(FMLoader,'{}'.format(action_index))
                if os.path.exists(action_dir):
                    target_read_path = os.path.join(action_dir,'0.jpg')
                    target_save_path = os.path.join(main_dir,'{}.jpg'.format(action_index))
                    print(action_index)
                    shutil.copy(target_read_path, target_save_path)

def Delete_vis(root, dir_indexes, move_file = False):
    # iterate root paths
    '''
    删除 tracking 和 Calibrate_transfer 两部分的 vis 图片
    '''
    for dir_index in dir_indexes:
        vis_dir = os.path.join(root, '{}/vis'.format(dir_index))
        action_indexes = os.listdir(vis_dir)

        for action_index in action_indexes:
            print('{} - {}'.format(dir_index, action_index))
            tracking_dir = os.path.join(vis_dir, '{}/tracking'.format(action_index))
            if os.path.isdir(tracking_dir):
                shutil.rmtree(tracking_dir)
            Calibrate_transfer_dir = os.path.join(vis_dir, '{}/Calibrate_transfer'.format(action_index))
            if os.path.isdir(Calibrate_transfer_dir):
                shutil.rmtree(Calibrate_transfer_dir)

if __name__ == '__main__':
    root = '/datanew/hwb/data/Football/SoftWare/'
    dir_indexes = [32]
    generate_main_imgs(root,dir_indexes)


