from CalibrateTransfer.class_set import *
import openpyxl
import numpy as np
import xlrd
import os

def getBenchmarks_from_txt(FILE_PATH):
    """get bench marks from txt file and arrange them properly!"""
    """
    FILE_PATH: The file path that save the txt file. 
    """
    result = []
    object_points = []
    img_points = []
    # open the benchmark_file and read all the object points and img points to the list
    with open(FILE_PATH, 'r') as f:
        for line in f:
            result.append(list(line.strip('\n').split('\t')))

    # seperate the object and img points from result list to their own list.
    points_number = 0
    for i in result:
        img_points.append(i[2])
        img_points.append(i[3])
        object_points.append(i[0])
        object_points.append(i[1])
        object_points.append(0)
        points_number += 1

    Object_points_np = np.array(object_points, np.dtype('float32')).reshape(points_number, 3)
    Imame_points_np = np.array(img_points, np.dtype('float32')).reshape(points_number, 2)

    # return points pair
    return Object_points_np, Imame_points_np

def get_action_data_from_txt(FILE_PATH):
    """get bench marks from txt file and arrange them properly!"""
    """
    FILE_PATH: The file path that save the txt file. 
    """
    teams = []
    player_numbers = []
    actions = []
    results = []
    T = []
    img_points = []
    object_points = []
    channels = []
    times = []
    data = []
    # open the benchmark_file and read all the object points and img points to the list
    with open(FILE_PATH, 'r') as f:
        for line in f:
            data.append(list(line.strip('\n').split('\t')))

    # seperate the object and img points from result list to their own list.
    points_number = 0
    for i in data:
        teams.append(i[0])
        player_numbers.append(i[1])
        actions.append(i[2])
        results.append(i[3])
        T.append(i[4])
        img_points.append(i[5])
        img_points.append(i[6])
        object_points.append(i[7])
        object_points.append(i[8])
        object_points.append(0)
        channels.append(i[9])
        times.append(i[10])
        points_number += 1

    object_points = np.array(object_points, np.dtype('float32')).reshape(points_number, 3).tolist()
    img_points = np.array(img_points, np.dtype('float32')).reshape(points_number, 2).tolist()

    # return points pair
    return teams,player_numbers,actions,results,T, img_points, object_points, channels, times

def write_CalibrateParameter_to_txt(FILE_NAME,calibrateParameter):
    """
    :param FILE_PATH:
    :return:
    """
    with open(FILE_NAME,'w+') as fp:
        fp.write(str(calibrateParameter.rotation_vector[0][0]) + '\n')
        fp.write(str(calibrateParameter.rotation_vector[1][0]) + '\n')
        fp.write(str(calibrateParameter.rotation_vector[2][0]) + '\n')
        fp.write(str(calibrateParameter.translation_vector[0][0]) + '\n')
        fp.write(str(calibrateParameter.translation_vector[1][0]) + '\n')
        fp.write(str(calibrateParameter.translation_vector[2][0]) + '\n')
        fp.write(str(calibrateParameter.cameraMatrix[0][0]) + '\n')
        fp.write(str(calibrateParameter.cameraMatrix[1][1]) + '\n')
        fp.write(str(calibrateParameter.cameraMatrix[0][2]) + '\n')
        fp.write(str(calibrateParameter.cameraMatrix[1][2]) + '\n')
        fp.write(str(calibrateParameter.distCoeffs[0][0]) + '\n')
        fp.write(str(calibrateParameter.distCoeffs[0][1]) + '\n')
        fp.write(str(calibrateParameter.distCoeffs[0][2]) + '\n')
        fp.write(str(calibrateParameter.distCoeffs[0][3]) + '\n')
        fp.write(str(calibrateParameter.distCoeffs[0][4]) )

def get_caliparameters_from_txt(FILE_PATH):
    data = []
    # open the benchmark_file and read all the object points and img points to the list
    with open(FILE_PATH, 'r') as f:
        for line in f:
            data.append(list(line.strip('\n').split('\t'))[0])

        calibrateParameter = CalibrateParameter(read_from_file=True)
        calibrateParameter.rotation_vector[0][0] = data[0]
        calibrateParameter.rotation_vector[1][0] = data[1]
        calibrateParameter.rotation_vector[2][0] = data[2]
        calibrateParameter.translation_vector[0][0] = data[3]
        calibrateParameter.translation_vector[1][0] = data[4]
        calibrateParameter.translation_vector[2][0] = data[5]
        calibrateParameter.cameraMatrix[0][0] = data[6]
        calibrateParameter.cameraMatrix[1][1] = data[7]
        calibrateParameter.cameraMatrix[0][2] = data[8]
        calibrateParameter.cameraMatrix[1][2] = data[9]
        calibrateParameter.cameraMatrix[2][2] = 1.0
        calibrateParameter.distCoeffs[0][0] = data[10]
        calibrateParameter.distCoeffs[0][1] = data[11]
        calibrateParameter.distCoeffs[0][2] = data[12]
        calibrateParameter.distCoeffs[0][3] = data[13]
        calibrateParameter.distCoeffs[0][4] = data[14]
    return calibrateParameter

def write_data_to_txt(FILE_NAME,teams,player_numbers,actions,results,T,Imame_points, Object_points, channels, times):
    """
    :param FILE_PATH:
    :return:
    """
    with open(FILE_NAME,'w+') as fp:
        for i in range(len(Imame_points)):
            fp.write(str(teams[i]) + '\t')
            fp.write(str(player_numbers[i]) + '\t')
            fp.write(str(actions[i]) + '\t')
            fp.write(str(results[i]) + '\t')
            fp.write(str(T[i]) + '\t')
            fp.write(str(Imame_points[i][0]) + '\t')
            fp.write(str(Imame_points[i][1]) + '\t')
            fp.write(str(Object_points[i][0]) + '\t')
            fp.write(str(Object_points[i][1]) + '\t')
            fp.write(str(channels[i]) + '\t')
            fp.write(str(times[i]) + '\n')

def read_xlsx(FILE_NAME):
    # 将整个数据文件.xlsx读进来，包含了上下半场
    book = xlrd.open_workbook(FILE_NAME)
    data = {}
    for i,sheet_name in enumerate(book.sheet_names()):
        # 读取每个工作表的名字
        sheet = book.sheets()[i]
        sub_data = []
        nrows = sheet.nrows
        for row_num in range(nrows):
            row_values = sheet.row_values(row_num)
            sub_data.append(row_values)
        # 按工作表的名字将数据存储在data字典里
        data[sheet_name] = sub_data
    return data

def write_xlsx(data,save_path):
    workbook = openpyxl.Workbook()
    sheet_num = len(data)
    for i,sheet_name in enumerate(data.keys()):
        # 按字典键值的名字生成对应的xlsx工作表
        if i == 0 :
            sheet = workbook.active
            sheet.title = sheet_name
        else:
            sheet = workbook.create_sheet(sheet_name)
        sub_data = data[sheet_name]
        for line in sub_data:
            sheet.append(line)
    workbook.save(save_path)


if __name__ == '__main__':
    # os.chdir('..')
    # root_path = '../westground/benchmarks'
    root_path = '../suzhou'
    data = read_xlsx(os.path.join(root_path,'test.xlsx'))
    write_xlsx(data,os.path.join(root_path,'save.xlsx'))

    # print()
    # FILE_PATH = os.path.join(root_path,'ch02_1.txt')
    # object_points, img_points = getBenchmarks_from_txt(FILE_PATH)


