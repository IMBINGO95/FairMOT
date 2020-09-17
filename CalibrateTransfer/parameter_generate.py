from .fb_cameraCalibrate import *
from .img_operation import *
from .file_operation import *
import json
import os
def parameter_generate(object_points, img_points, img_width,img_height):
    '''
    generete parameters and return imgs to evaluate the quality of these parameters!
    :param object_points: 
    :param img_points: 
    :param img: 
    :param field_width_start: 
    :param field_width_end: 
    :param field_length_start: 
    :param field_length_end: 
    :param step_length:
    :return: 
    '''
    calibrateParameter = fb_cameraCalibrate(object_points, img_points, img_width, img_height) # get the calibrateParamter.
    parameter_dict = CalibrateParameter_To_dict(calibrateParameter) # change parameter from CalibrateParameter class type to list type.

    return parameter_dict,calibrateParameter

def sort_points(index,points):
    Points = []
    for i in index:
        Points.append(points[i-1])
    return Points

if __name__ == '__main__':
    # the file root_path of corresponding field.
    root_path = './westground'
    root_path = './sugar_box'
    # how many channels does this field installed
    # channels = {'ch01':4,'ch02':1,'ch03':1 , 'ch04':1 , 'ch05':1 , 'ch06':1, 'ch07':1}
    channels = {'ch01':1,'ch02':1,'ch03':1 , 'ch04':1}
    # channels = {'ch01': 1 }
    data = {}  # 要保存这个data成json文件，软件把这个data作为标定文件输入。
    field_name = 'sugar_box'
    data['field_name'] = field_name
    data['params'] = {}
    data_save_path = os.path.join(root_path, field_name + '.json')
    field_length, field_width= 56, 46.5 # 场地的长宽
    # 逐个通道进行处理
    for channel in channels:
        camera_num = channels[channel] # 每个通道的摄像头由几个相机拼接成，通常是一个，全景相机是四个。
        img_path = os.path.join(root_path, 'imgs', channel + '.jpg') #图片的保存地址，图片的命名要按照规则来。
        img = cv2.imread(img_path)
        # 使用灰色填充图片区域,默认为黑色
        filed_2D_view = np.zeros((int((field_length + 5) * 15), int((field_length + 10) * 15), 3), np.uint8)
        filed_2D_view.fill(123)
        calibrateParameter = {}
        img_width = 2704
        img_height = 1520
        for j in range(1,camera_num+1):
            txt_path = os.path.join(root_path, 'benchmarks', channel + '_' + str(j) + '.txt')
            '''check the img_points, figure out if it is in its correspongding area.'''
            object_points, img_points = getBenchmarks_from_txt(txt_path)
            '''The area that we want to check,check the object_points and its corresponding area in the img'''
            field_width_start = min(object_points[:, 0])
            field_width_end = max(object_points[:, 0])
            field_length_start = min(object_points[:, 1])
            field_length_end = max(object_points[:, 1])
            step_length = 0.2

            '''check the img_points, figure out if they are correct and in their correspongding area.'''
            img = check_points(img,img_points)
            # index = [i for i in range(1,len(object_points))]
            # out = [1,2,3,4,5,6,17,18,19,31]
            # out = []
            # for i in out :
            #     index.remove(i)
            # Object_points = sort_points(index, object_points)
            # Img_points = sort_points(index,img_points)
            '''calculate the parameters of each camera.'''
            parameter_dict, sub_Parameter = parameter_generate(object_points, img_points,img_width, img_height)
            section_name = 'section{}'.format(j)
            calibrateParameter[section_name] = parameter_dict

            '''check the object_points, figure out if they are correct and in their correspongding area.
            and transfer img_points to object view, and to visulize the error betweent them.'''
            filed_2D_view = visulize_calibrate_error(filed_2D_view, object_points, img_points, sub_Parameter, field_width)

            # 将参数的效果以可视化的效果展现出来。
            if channel == 'ch01':
                img = draw_points_on_img_ch01(img, field_width_start, field_width_end, field_length_start,
                                              field_length_end, step_length, sub_Parameter, section_num=j)
            else:
                img = draw_points_on_img(img, field_width_start, field_width_end, field_length_start,field_length_end, step_length, sub_Parameter)

            txt_save_path = os.path.join(root_path, channel + '_{}.txt'.format(j))
            write_CalibrateParameter_to_txt(txt_save_path, sub_Parameter)
        # save the field parameters to json file
        data['params'][channel] = calibrateParameter

        save_path = os.path.join(root_path, 'imgs', channel + '_1.jpg')

        cv2.imwrite(save_path, img)

        save_path = os.path.join(root_path, 'imgs', channel + '_2.jpg')
        cv2.imwrite(save_path, filed_2D_view)

    with open(data_save_path,'w') as f:
        json.dump(data,f)
    print()



