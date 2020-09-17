'''
Limit the detection range to the range we are interested in.
'''
import cv2

def calculate_line(point1,point2,mode):
    '''
    according to two points to calculate the k and b (y = kx + b)
    :param point1:
    :param point2:
    :param mode:
    :return:
    '''
    k = (point2[1]-point1[1])/(point2[0]-point1[0])
    b = point1[1] - k*point1[0]
    return [k,b,mode]

def generate_line_param():
    '''output params [k,b,mode] that
     are used to describe the lines that specifies the site range.'''
    point_pairs = []
    # line1
    point_pairs.append([(2953,1207), (2009,1170), -1])
    #  1 means the point needs to beyong the line aspect to pixel_point(x=0,y=0)
    # -1 means the point needs to below the line aspect to pixel_point(x=0,y=0)
    # line2
    point_pairs.append([(2009,1170), (1062,1031), -1])
    # line3
    point_pairs.append([(1062,1031), (84,824), -1])
    # line4
    point_pairs.append([(84,824), (1680,248), 1])
    # line5
    point_pairs.append([(1680,248), (2291,197), 1])
    # line6
    point_pairs.append([(2291,197), (2941,164), 1])
    # line7
    point_pairs.append([(2941,164), (3636,175), 1])
    # line8
    point_pairs.append([(3636,175), (4283,209), 1])
    # line9
    point_pairs.append([(4283,209), (4890,404), 1])
    # line10
    point_pairs.append([(4890,404), (5883,772), 1])
    # line11
    point_pairs.append([(5883,772), (4933,983), -1])
    # line12
    point_pairs.append([(4933,983), (3946,1151), -1])
    # line13
    point_pairs.append([(3946,1151), (2953,1207), -1])
    parms = []

    for point_pair in point_pairs:
        parm = calculate_line(point_pair[0], point_pair[1], point_pair[2])
        parms.append(parm)
        # cv2.line(img1,point_pair[0] , point_pair[1], color=[255, 0, 255], thickness=3)
    return parms

def area_constrict(box,parms,mode):
    '''

    :param box: contains input box
    :param parms: contains [k,b,mode] that
            are used to describe the lines that specifies the site range.
    :param mode:mode = 0 means box = [xl,yl,xr,yr]
                mode = 1 means box = [xl,yl,weight,height]
    :return:
    '''
    # compute the bottom center of this box.
    if mode == 0:
        x = (box[0] + box[2]) / 2
        y = box[3]
    else:
        x = box[0] + box[2] / 2
        y = box[1] + box[3]

    # Determine if this box is within range of interest?
    results = []
    for parm in parms:
        # determin y-kx-b >=0 or <0 ?
        result = y - parm[0] * x - parm[1]
        if result * parm[2]>= 0 :
            results.append('True')
        else:
            results.append('False')
    if 'False' in results:
        return  False
    else :
        return True