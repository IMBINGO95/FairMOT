from CalibrateTransfer.class_set import *
import cv2

def fb_cameraCalibrate_ch01(FILE_PATH_Section1,FILE_PATH_Section2,FILE_PATH_Section3,FILE_PATH_Section4, img_width_Flag1, img_height_Flag1):
    """
    Just as the function name means, This function is to calibrate the camera parameters!
    But this function is specified for Flag1, because Flag1 camera is combined by four seperate camera.
    So it needs four seperate calibrate parameters for different sections.
    :param FILE_PATH_Section1:
    :param FILE_PATH_Section2:
    :param FILE_PATH_Section3:
    :param FILE_PATH_Section4:
    :param img_width:
    :param img_height:
    :return:
    """
    # get parameters for different sections of Flag1 camera.
    calibrateParameter_section1 = fb_cameraCalibrate(FILE_PATH_Section1, img_width_Flag1, img_height_Flag1)
    calibrateParameter_section2 = fb_cameraCalibrate(FILE_PATH_Section2, img_width_Flag1, img_height_Flag1)
    calibrateParameter_section3 = fb_cameraCalibrate(FILE_PATH_Section3, img_width_Flag1, img_height_Flag1)
    calibrateParameter_section4 = fb_cameraCalibrate(FILE_PATH_Section4, img_width_Flag1, img_height_Flag1)


    calibrateParameter_Flag1 = CalibrateParameter_Flag1(calibrateParameter_section1,calibrateParameter_section2,calibrateParameter_section3,calibrateParameter_section4)

    return calibrateParameter_Flag1

def fb_cameraCalibrate(object_points, img_points, img_width, img_height):
    '''
    Just as the function name means, This function is to calibrate the camera parameters!
    :param object_points:  benchmarks in list type!
    :param img_points:
    :param img_width:
    :param img_height:
    :return: calibrateParameter, param
    '''
    '''if there object_points and img_points and lists then change them to numpy array!'''
    object_points = np.array(object_points)
    img_points = np.array(img_points,dtype=np.float32)
    # img_points = img_points.astype(float)  # transfer to int type
    # get the cameraCalibrate parameter.
    ret, cameraMatrix, distCoeffs, rotation_vector, translation_vector = cv2.calibrateCamera([object_points],
                                                                                             [img_points],
                                                                                             (img_width, img_height),
                                                                                             None,
                                                                                             None)
    # save the parameter on class CalibrateParameter.
    calibrateParameter = CalibrateParameter(rotation_vector[0], translation_vector[0], cameraMatrix, distCoeffs)

    return calibrateParameter

