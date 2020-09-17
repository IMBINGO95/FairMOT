import numpy as np
class CalibrateParameter():
    """相机的标定参数"""
    def __init__(self,rotation_vector = None, translation_vector = None, cameraMatrix = None, distCoeffs = None, read_from_file = False):
        ''' If read_from_file == True, then creat a empty class!'''
        if read_from_file :
            self.rotation_vector = np.zeros((3, 1), dtype='float64')  # 大写是类！
            self.translation_vector = np.zeros((3, 1), dtype='float64')  # 大写是类！
            self.cameraMatrix = np.zeros((3, 3), dtype='float64')  # 大写是类！
            self.distCoeffs = np.zeros((1, 5), dtype='float64')  # 大写是类！
        else:
            self.rotation_vector = rotation_vector
            self.translation_vector = translation_vector
            self.cameraMatrix = cameraMatrix
            self.distCoeffs = distCoeffs

class CalibrateParameter_Flag1():
    """Wide-angle camera calibration parameters for Flag1
    But this Class is specified for Flag1, because Flag1 camera is combined by four seperate camera.
    So it needs four seperate calibrate parameters for different sections.
    """
    def __init__(self,Calib_Section1,Calib_Section2,Calib_Section3,Calib_Section4):
        self.Calib_Section1 = Calib_Section1
        self.Calib_Section2 = Calib_Section2
        self.Calib_Section3 = Calib_Section3
        self.Calib_Section4 = Calib_Section4


class worldcoor():
    """定义世界坐标,需要输入x,y的值，默认z = 0 。"""
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.z = 0

def CalibrateParameter_To_dict(calibrateParameter):
    '''
    change CalibrateParameter class to dictionay type!
    :param calibrateParameter: parameter in CalibrateParameter class type.
    :return: parameter in dictionay type.
    '''
    parameter = {}
    parameter['rotation'] = {}
    parameter['rotation']['alpha'] = calibrateParameter.rotation_vector[0][0]
    parameter['rotation']['beta'] = calibrateParameter.rotation_vector[1][0]
    parameter['rotation']['gama'] = calibrateParameter.rotation_vector[2][0]
    parameter['translation'] = {}
    parameter['translation']['tx'] = calibrateParameter.translation_vector[0][0]
    parameter['translation']['ty'] = calibrateParameter.translation_vector[1][0]
    parameter['translation']['tz'] = calibrateParameter.translation_vector[2][0]
    parameter['intrinsic'] = {}
    parameter['intrinsic']['fx'] = calibrateParameter.cameraMatrix[0][0]
    parameter['intrinsic']['fy'] = calibrateParameter.cameraMatrix[1][1]
    parameter['intrinsic']['u'] = calibrateParameter.cameraMatrix[0][2]
    parameter['intrinsic']['v'] = calibrateParameter.cameraMatrix[1][2]
    parameter['distortion'] = {}
    parameter['distortion']['k1'] = calibrateParameter.distCoeffs[0][0]
    parameter['distortion']['k2'] = calibrateParameter.distCoeffs[0][1]
    parameter['distortion']['p1'] = calibrateParameter.distCoeffs[0][2]
    parameter['distortion']['p2'] = calibrateParameter.distCoeffs[0][3]
    parameter['distortion']['k3'] = calibrateParameter.distCoeffs[0][4]

    return parameter

def dict_To_CalibrateParameter(parameter):
    '''
    change dictionay type to CalibrateParameter type
    :param parameter:
    :return:
    '''
    calibrateParameter = CalibrateParameter(read_from_file=True)
    calibrateParameter.rotation_vector[0][0] = parameter['rotation']['alpha']
    calibrateParameter.rotation_vector[1][0] = parameter['rotation']['beta']
    calibrateParameter.rotation_vector[2][0] = parameter['rotation']['gamma']
    calibrateParameter.translation_vector[0][0] = parameter['translation']['tx']
    calibrateParameter.translation_vector[1][0] = parameter['translation']['ty']
    calibrateParameter.translation_vector[2][0] = parameter['translation']['tz']
    calibrateParameter.cameraMatrix[0][0] = parameter['intrinsic']['fx']
    calibrateParameter.cameraMatrix[1][1] = parameter['intrinsic']['fy']
    calibrateParameter.cameraMatrix[0][2] = parameter['intrinsic']['u']
    calibrateParameter.cameraMatrix[1][2] = parameter['intrinsic']['v']
    calibrateParameter.cameraMatrix[2][2] = 1.0
    calibrateParameter.distCoeffs[0][0] = parameter['distortion']['k1']
    calibrateParameter.distCoeffs[0][1] = parameter['distortion']['k2']
    calibrateParameter.distCoeffs[0][2] = parameter['distortion']['p1']
    calibrateParameter.distCoeffs[0][3] = parameter['distortion']['p2']
    calibrateParameter.distCoeffs[0][4] = parameter['distortion']['k3']

    return calibrateParameter
