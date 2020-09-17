import cv2
import numpy as np

def sort_by_point(result, reference_point, IoUthreshold=0.5,input_index=None):

    new_bbox_det = np.vstack(result[1])  # 输入的是[xl,yl,w,h]
    new_bbox_det[:,2:] += new_bbox_det[:,0:2] # 改成[xl,yl,xr,yr]
    new_bbox_det_bc = np.zeros((new_bbox_det.shape[0],2)) # 建立一个底部中心矩阵，用来计算距离
    new_bbox_det_bc[:,0] = 0.5*(new_bbox_det[:,0] + new_bbox_det[:,2]) # 底部中心的x
    new_bbox_det_bc[:,1] = new_bbox_det[:,3] # 底部中心的y

    residual = new_bbox_det_bc - np.array([reference_point])
    residual = np.square(residual)
    res_sum = np.sqrt(np.sum(residual,axis=1))  # calculate the Sta ndard deviation
    res_min = np.min(res_sum)

    # 针对两种不同的输入，做相应的更改，一个有输入ID，一个没有
    index = int(np.where(res_sum==res_min)[0])

    ids = result[2] # 用来区分两种模式
    # ids == False 代表用于检测网络的筛选

    if ids:
        target_id = ids[index]
    else:
        target_id = index

    '''target is the nearlest box to the img_point '''
    target = new_bbox_det[index,:]
    '''set margin to  constraint the distance between reference point and target box.'''
    # target_width = round(target[2]-target[0])
    target_width = round(target[2]-target[0]) # target box 的宽
    # if res_ m in > min(margin, target_width):
    if res_min > target_width:
        # print('res_min = {:.2f} bigger target_box_width = {:.2f}, index = {}'.format(res_min, target_width,input_index))
        return None,None
    # else:
    #  	print('res_min = {:.2f} smaller target_box_width = {:.2f}'.format(res_min, target_width))

    '''calculate the IoUs between target box and the other boxes.'''
    length, _ = new_bbox_det.shape
    # length = len(ids)
    if length > 1:
        new_bbox_det = np.delete(new_bbox_det,index,axis=0)
        ixmin = np.maximum(new_bbox_det[:,0],target[0])
        iymin = np.maximum(new_bbox_det[:,1],target[1])
        ixmax = np.minimum(new_bbox_det[:,2],target[2])
        iymax = np.minimum(new_bbox_det[:,3],target[3])
        iw = np .maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((target[2]-target[0] +1) * (target[3] - target[1] +1)
               + (new_bbox_det[:,2]-new_bbox_det[:,0] + 1)*(new_bbox_det[:,3]-new_bbox_det[:,1] + 1)
               - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        if ovmax > IoUthreshold:
            return None,None

    box = target.astype(np.dtype('int64')).tolist()
    # The rectangular img of the target person
    # The n ew reference point is the bottom center of the target bbox .
    new_reference_point = (int(0.5*(box[0]+box[2])),box[3])
    return new_reference_point,target_id

if __name__ == '__main__':
    bboxes = [np.array([879.5273032313518,403.4230331269775,50.47936895587865,116.77033938969033]),
              np.array([570.8644439850867,280.3942436089609,54.87601045753755,116.51562743364121]),
              np.array([843.0158720696529,239.04089414756467,42.565160801793006,97.5358089245354]),
              np.array([-3.557812338971445,210.16708296348014,46.930830104392626,104.46519722432618]),
              np.array([473.4352707263196,35.95088777691288,29.631849362846342,69.33894870374692]),
              np.array([56.159225948431526,124.99089818338777,37.19768653553596,90.62527880909323]),
              np.array([58.53890490766025,458.6884215692329,29.227418744866704,116.6273027549012]),
              np.array([940.1416324646867,0.2254591082956452,38.96061000546591,49.877468700547446]),
              np.array([474.5373356020795,193.1245219560403,45.09583947748563,96.72719549769175]),
              np.array([831.7861086857376,-0.6187301525027493,29.19942916389558,66.22413980668051])]
    ids = [i for i in range(10)]
    result = [0,bboxes,ids]
    reference_point = [544,408]
    # new_reference_point, target_id = sort_by_point(result,reference_point)
    # print(target_id)
    ids = np.random.randint(0,10,size=(10,10))
    b = np.where(ids==2)

    print(b)




