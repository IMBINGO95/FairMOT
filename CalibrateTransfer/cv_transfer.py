import cv2
import numpy as np

def object_To_pixel(object_points, calibrateParameter):
    """
    :param object_points is [N,3] numpy array, N is the points number.
    :param calibrateParameter:is class of CalibrateParameter():
    :return:
    """

    imagePoints, jac = cv2.projectPoints(object_points,
                                         calibrateParameter.rotation_vector,
                                         calibrateParameter.translation_vector,
                                         calibrateParameter.cameraMatrix,
                                         calibrateParameter.distCoeffs)

    return imagePoints

def transform_2d_to_3d_for_channel1(img_point,calibrateParameter_all):
    '''
        判断像素坐标属于Flag1下的那一块区域，从而判断应该采用那一组标定参数。
        :param img_point: which must be a [1,2] numpy array
                or a of [1,2] list
        :param calibrateParameter_Flag1: is class of CalibrateParameter_Flag1()
        :return:
        '''
    # x_of_line_between_section1_and_section2 = 1280;
    x_1_2 = 1280
    # x_of_line_between_section2_and_section3 = 3060;
    x_2_3 = 3060
    # x_of_line_between_section3_and_section4 = 4108;
    x_3_4 = 4108

    if (img_point[0] >= 0 and img_point[0] < x_1_2):
        calibrateParameter = calibrateParameter_all[0]
    elif (img_point[0] >= x_1_2 and img_point[0] < x_2_3):
        calibrateParameter = calibrateParameter_all[1]
    elif (img_point[0] >= x_2_3 and img_point[0] < x_3_4):
        calibrateParameter = calibrateParameter_all[2]
    elif (img_point[0] >= x_3_4 and img_point[0] < 5950):
        calibrateParameter = calibrateParameter_all[3]

    object_point = transform_2d_to_3d(img_point, calibrateParameter.cameraMatrix, calibrateParameter.distCoeffs,
                               calibrateParameter.rotation_vector,
                               calibrateParameter.translation_vector, world_z=0)
    return object_point

def transform_2d_to_3d(point2d,mtx,dist,rvec,tvec,world_z):
    '''

    :param point2d: the img_point that we want to project back to real world.
    :param mtx:
    :param dist:
    :param rvec:
    :param tvec:
    :param world_z:  always set to 0
    :return:
    '''
    # handling distortion problems.
    # print(mtx,dist,rvec,tvec)
    point = np.array(point2d,dtype='float32')[np.newaxis, np.newaxis, :]
    point2d= cv2.undistortPoints(point, mtx, dist, P=mtx)
    point2d = list(point2d[0, 0])
    # print('undistort:',point2d)
    R,_=cv2.Rodrigues(rvec)
    screen_coordinates=np.array([point2d[0],point2d[1],1])
    left_side_mat=np.linalg.inv(R).dot(np.linalg.inv(mtx)).dot(screen_coordinates)
    right_side_mat=np.linalg.inv(R).dot(tvec)
    s=(world_z+right_side_mat[2,0])/left_side_mat[2]
    wc_point=np.linalg.inv(R).dot((s*np.linalg.inv(mtx).dot(screen_coordinates))-tvec.reshape(3,))
    return wc_point

def transform_2d_to_3d_multi(point2d,mtx,dist,rvec,tvec,world_z):
    '''

    :param point2d: the img_point that we want to project back to real world.
    :param mtx:
    :param dist:
    :param rvec:
    :param tvec:
    :param world_z:  always set to 0
    :return:
    '''
    # handling distortion problems.
    # print(mtx,dist,rvec,tvec)
    point = np.array(point2d,dtype='float32')[np.newaxis, :]
    point2d= cv2.undistortPoints(point, mtx, dist, P=mtx)
    point2d = list(point2d[0, 0])
    # print('undistort:',point2d)
    R,_=cv2.Rodrigues(rvec)
    screen_coordinates=np.array([point2d[0],point2d[1],1])
    left_side_mat=np.linalg.inv(R).dot(np.linalg.inv(mtx)).dot(screen_coordinates)
    right_side_mat=np.linalg.inv(R).dot(tvec)
    s=(world_z+right_side_mat[2,0])/left_side_mat[2]
    wc_point=np.linalg.inv(R).dot((s*np.linalg.inv(mtx).dot(screen_coordinates))-tvec.reshape(3,))
    return wc_point


def transform_2d_to_3d_2(point2d,mtx,dist,rvec,tvec,world_z):
    point2d = list(cv2.undistortPoints(np.array(point2d)[np.newaxis, np.newaxis, :], mtx, dist, P=mtx)[0, 0])
    R,_=cv2.Rodrigues(rvec)
    extrinsM=np.concatenate((R,tvec),axis=1)
    P=mtx.dot(extrinsM)
    p11=P[0,0]
    p12=P[0,1]
    p14=P[0,3]
    p21=P[1,0]
    p22=P[1,1]
    p24=P[1,3]
    p31=P[2,0]
    p32=P[2,1]
    p34=P[2,3]
    homographyMatrix=np.array([[p11,p12,p14],[p21,p22,p24],[p31,p32,p34]])
    invh=np.linalg.inv(homographyMatrix)
    screen_coordinates=np.array([point2d[0],point2d[1],1])
    point3d_w = invh.dot(screen_coordinates)
    w=point3d_w[2]
    wc_point=cv2.divide(point3d_w,w)
    return wc_point

def updata_img_point(reference_point, new_reference_point, img_point):
    '''
    update img_point by distance between old and new reference point
    :param reference_point:
    :param new_reference_point:
    :param img_point:
    :return:
    '''
    delet_x = new_reference_point[0] - reference_point[0]
    delet_y = new_reference_point[1] - reference_point[1]
    img_point = (img_point[0] + delet_x, img_point[1] + delet_y)
    return img_point