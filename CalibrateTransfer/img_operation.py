import cv2
import numpy as np
import copy
from CalibrateTransfer.cv_transfer import *


def draw_points_on_img(img,filed_width_start, filed_width_end, filed_length_start,filed_length_end, step_length, calibrateParameter):
	'''draw points on the img'''
	'''
	:param img_path:
	:param filed_width_start:  the start point of width
	:param filed_width_end:    the end point of width
	:param filed_length_start: the start point of length
	:param filed_length_end:   the end point of length
	:param step_length:        the point gap between points
	:param calibrateParameter:
	:return:
	'''
	# creat a empty worldpoints
	worldpoint = np.zeros([1,3])

	#
	list1 = list(np.arange(filed_width_start, filed_width_end, step_length))
	list2 = list(np.arange(filed_length_start,filed_length_end, step_length))
	# A for loop
	for i in list1:
		for j in list2:

			worldpoint[0,0] = i
			worldpoint[0,1] = j
			#print('i=',i,'j = ',j)

			imgpoint = object_To_pixel(worldpoint,calibrateParameter)

			x = int(imgpoint[0][0][0])
			y = int(imgpoint[0][0][1])
			#print(x,y)

			# draw points on the img
			if (x>= 0 and x<= 4000 and y >= 0 and y <= 3000):
				cv2.circle(img,(x, y ),2,(0,0,255),-1)

	return img

def draw_points_on_img_ch01(img, field_width_start, field_width_end,field_length_start,field_length_end, step_length, sub_Parameter, section_num):
	'''draw points on the img'''
	'''
	:param filed_width_start:  the start point of width
	:param filed_width_end:    the end point of width
	:param filed_length_start: the start point of length
	:param filed_length_end:   the end point of length
	:param step_length:        the point gap between points
	:param Parameter: is class of CalibrateParameter():
	:return: img
	'''
	# creat a empty worldpoints
	worldpoint = np.zeros([1,3])
	list1 = list(np.arange(field_width_start, field_width_end, step_length))
	list2 = list(np.arange(field_length_start,field_length_end, step_length))
	# A for loop
	Flag = False  # if Flag == True ，then draw a points on the img.
	for i in list1:
		for j in list2:
			if section_num == 1:
				if ((j - 0.8289 * i <= -22.2643) and (i >= 0)):
					Flag = True
			elif section_num == 2 :
				if ((j - 0.8289 * i > -22.2643) and (j + 0.0452 * i <= 54.76)):
					Flag = True
			elif section_num == 3:
				if ((j + 0.0452 * i > 54.76 ) and ( j + 0.4926 * i <= 93.96)):
					Flag = True
			else:
				if ((j + 0.4926 * i > 93.96) and j < 101.88):
					Flag = True
			if Flag == True:
				worldpoint[0,0] = i
				worldpoint[0,1] = j
				#print('i=',i,'j = ',j)
				imgpoint = object_To_pixel(worldpoint,sub_Parameter)
				x = int(imgpoint[0][0][0])
				y = int(imgpoint[0][0][1])
				# draw points on the img
				cv2.circle(img,(x, y ),1,(0,0,255),-1)
			Flag = False
	return img

def check_points(img,img_points=None,object_points=None,field_width_end= None,Flag=0):
	if Flag == 0:
		'''check the img_points, figure out if they are correct and in their correspongding area.'''
		for i, point in enumerate(img_points):
			cv2.circle(img, (point[0], point[1]), 10, (255, 0, 0), -1)
			cv2.putText(img, '{}'.format(i + 1), org=(point[0], point[1]), fontFace=cv2.FM_8POINT,
						fontScale=8,
						color=(255, 0, 0))
	else:
		'''check the object_points, figure out if they are correct and in their correspongding area.'''
		for i, point in enumerate(object_points):
			cv2.circle(img, (int((field_width_end - point[0] + 10) * 10), int(((point[1] + 10) * 10))), 10,
					   (255, 0, 0), -1)
			cv2.putText(img, '{}'.format(i + 1),
						org=(int((field_width_end - point[0] + 10) * 10), int(((point[1] + 10) * 10))),
						fontFace=cv2.FM_8POINT, fontScale=8, color=(255, 0, 0))
	return img

def visulize_calibrate_error(img,object_points,img_points,calibrateParameter,field_width):
	'''check the object_points, figure out if they are correct and in their correspongding area.
				and transfer img_points to object view, and to visulize the error betweent them.'''
	for i, point in enumerate(object_points):
		cv2.circle(img, (int((field_width - point[0] + 10) * 10), int(((point[1] + 10) * 10))), 10, (255, 0, 0),
				   -1)
		cv2.putText(img, '{}'.format(i + 1),
					org=(int((field_width - point[0] + 10) * 10), int(((point[1] + 10) * 10))),
					fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0))

	for i, point in enumerate(img_points):
		'''transfer img_points to object view'''
		point = transform_2d_to_3d(point, calibrateParameter.cameraMatrix, calibrateParameter.distCoeffs,
								   calibrateParameter.rotation_vector,
								   calibrateParameter.translation_vector, world_z=0)
		cv2.circle(img, (int((field_width - point[0] + 10) * 10), int(((point[1] + 10) * 10))), 10, (0, 0, 255),
				   -1)
		cv2.putText(img, '{}'.format(i + 1),
					org=(int((field_width - point[0] + 10) * 10), int(((point[1] + 10) * 10))),
					fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))
	return img


def GenerateRect(img_point,Output_size,bias,width,height):
	'''
	Given the img point and img size ,determine the region of the targer area.
	if pixel exits in this img return True and the corresponding area!
	else return false!
	:param img_point:
	:param Output_size:
	:param bias:
	:param width:
	:param height:
	:return:
	'''
	Message = [False,()]
	x = img_point[0]
	y = img_point[1]
	output_x = Output_size[0]
	output_y = Output_size[1]

	'''pixel point out of img'''
	if x < 0 or x >  width or y < 0 or y > height:
		Message[0] = False

	# pixel at the corner
	# upper left corner
	elif ( x <=  0.5 * output_x) and (y <= output_y - bias):
		Message[0] = True
		Message[1] = (0, 0, output_x,output_y)

	# upper right corner
	elif( x >=  (width - 0.5 * output_x)) and (y <= output_y - bias):
		Message[0] = True
		Message[1] = (width - output_x , 0, output_x, output_y)

	# lower left corner
	elif (x <= 0.5 * output_x) and (y >= (height - bias)):
		Message[0] = True
		Message[1] = (0, height-output_y, output_x, output_y)

	# lower right corner
	elif ( x >=  (width - 0.5 * output_x)) and (y >= (height - bias)):
		Message[0] = True
		Message[1] = (width - output_x, height - output_y , output_x, output_y)

	# pixel at the img edge ( upper left right )
	# upper
	elif y <= output_y - bias:
		Message[0] = True
		Message[1] = (x - 0.5 * output_x, 0 , output_x, output_y)

	# left
	elif x <= 0.5 * output_x:
		Message[0] = True
		Message[1] = (0, y - output_y + bias, output_x, output_y)

	# right
	elif x >= width - 0.5 * output_x:
		Message[0] = True
		Message[1] = (width - output_x, y - output_y + bias, output_x, output_y)

	# lower
	elif y >= height - bias:
		Message[0] = True
		Message[1] = (x - 0.5 * output_x, y - output_y, output_x, output_y)
	# other situation
	else:
		Message[0] = True
		Message[1] = (x - 0.5 * output_x, y - output_y + bias, output_x, output_y)

	return Message




def _to_color(indx, base):
	""" return (b, r, g) tuple"""
	base2 = base * base
	b = 2 - indx / base2
	r = 2 - (indx % base2) / base
	g = 2 - (indx % base2) % base
	return b * 127, r * 127, g * 127

def boxes_refine(allboxes_per_frame,width_thr,heigth_thr):
	'''
	refine the boxes detected by CNN
	:param allboxes_per_frame: allboxes_per_frame[0] in the shape of [frame_num,xl,yl,xr,yr,confidence,label]
	:param area_thr: if the box area bigger than area_thr,then discard it.
	:param confidence_thr: if the box confidence smaller than confidence_thr,then discard it.
	:return: allboxes_per_frame
	'''
	new_boxes = []
	for i,box in enumerate(allboxes_per_frame):
		area = (box[1] - box[3]) * (box[2] - box[4])
		if (box[3]-box[1]> width_thr) or (box[4]-box[2] > heigth_thr) or (int(box[6]) != 1):
			continue
		else:
			# transform [xl,yl,xr,yr] to [xl,yl,w,h]
			box[3] = box[3] - box[1]
			box[4] = box[4] - box[2]
			new_boxes.append(box)

	# new_bbox_det.resize(int(length), 6)
	return new_boxes

def divide_range(Ori_img_W, Ori_img_H, section_num, mode=0):
	'''

	:param Ori_img_W:
	:param Ori_img_H:
	:param section_num: divide the weight/height to (%s section_num) parts.
	:param mode: mode=0 to divide weight, mode=1 to divide height
	:return:
	'''
	if mode==0:
		section = []
		U_rate = 0.2
		# n-1 incomplete(1-U_rate) plus a complete one
		part_img_w = int(Ori_img_W / ((section_num - 1) * (1 - U_rate) + 1))
		w_range = []
		for i in range(section_num):
			part_ = [0, Ori_img_H, int(i * part_img_w * (1 - U_rate)), int(i * part_img_w * (1 - U_rate)) + part_img_w]
			if i == section_num - 1:
				part_[3] = Ori_img_W
			section.append(part_)
			w_range.append(part_[2])
			w_range.append(part_[3])
		box_range = []
		middle = []
		w_range.sort()
		middle.append(w_range[0])
		for i in range(1, len(w_range) - 1, 2):
			w_ = int((w_range[i] + w_range[i + 1]) / 2)
			middle.append(w_)
		middle.append(w_range[-1])
		for i in range(section_num):
			box_range.append([middle[i], middle[i + 1]])
		return section,box_range,part_img_w
	else:
		section = []
		U_rate = 0.3
		part_img_h = int(Ori_img_H / ((section_num - 1) * (1 - U_rate) + 1))
		h_range = []
		for i in range(section_num):
			part_ = [int(i * part_img_h * (1 - U_rate)), int(i * part_img_h * (1 - U_rate)) + part_img_h, 0, Ori_img_W]
			if i == section_num - 1:
				part_[1] = Ori_img_H
			section.append(part_)
			h_range.append(part_[0])
			h_range.append(part_[1])
		box_range = []
		middle = []
		h_range.sort()
		middle.append(h_range[0])
		for i in range(1, len(h_range) - 1, 2):
			h_ = int((h_range[i] + h_range[i + 1]) / 2)
			middle.append(h_)
		middle.append(h_range[-1])
		for i in range(section_num):
			box_range.append([middle[i], middle[i + 1]])
		return section, box_range,part_img_h


def draw_detectionv1(im, bboxes,thr,visualization=False):
	'''If there is no bbox whose score is bigger than threshold,then return None'''
	# bboxes format [x_l,y_l,x_r,y_r,score]
	indexs = np.where(bboxes[:,4] >= thr)

	if len(indexs[0]) != 0:
		thick = 4  # line thick
		imgcv = np.copy(im)
		new_bbox_det = []
		new_bbox_det = np.array(new_bbox_det)

		colors = (125, 0, 125)
		for index in indexs[0]:
			box = bboxes[index]
			if visualization == True:
				cv2.rectangle(imgcv,
							  (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
							  colors, thick)
				mess = '%.3f' % (box[4])
				cv2.putText(imgcv, mess, (int(box[0]), int(box[1] - 7)), 2, 2, (20,9,77), 1)



		return (imgcv,True,bboxes[indexs[0]])
	else:
		return (None,False,None)

'''can not delete this code'''
# def draw_detection(im, bboxes, scores, cls_inds, fps, thr=10):
#     imgcv = np.copy(im)
#     h, w, _ = imgcv.shape
#     new_bbox_det = []
#     new_bbox_det = np.array(new_bbox_det)
#     for i, box in enumerate(bboxes):
#         area = (box[0] - box[2]) * (box[1] - box[3])
#         if (area > 500*300) or (scores[i] < thr) or (int(cls_inds[i]) != 1):
#             continue
#         cls_indx = int(cls_inds[i])
#         box = [int(_) for _ in box]
#         thick = int((h + w) / 300)
#         cv2.rectangle(imgcv,
#                       (box[0], box[1]), (box[2], box[3]),
#                       colors[cls_indx], thick)
#         mess = '%s: %.3f' % (labels[cls_indx], scores[i])
#         cv2.putText(imgcv, mess, (box[0], box[1] - 7),
#                     0, 1e-3 * h, colors[cls_indx], thick // 3)
#         if fps >= 0:
#             cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15), 0, 2e-3 * h, (255, 255, 255), thick // 2)
#
#         sub_new_det = np.array((box[0] , box[1] , box[2] , box[3], cls_indx))
#         new_bbox_det = np.insert(new_bbox_det, 0, values=sub_new_det, axis=0)
#         length = new_bbox_det.size / 5
#
#     new_bbox_det.resize(int(length),5)
#     return imgcv,new_bbox_det


def sort_by_point(im, bboxes, scores, reference_point, thr=10, visualization = False, margin=480, IoUthreshold=0.5):
	"""
	If there is no bbox which score is bigger than threshold,
	or if the distance between reference point and target box is bigger than margin
	or if the IoU of the target box is bigger than IoUthreshold
	then return (im, None, [])
	else return (imgcv,sub_img,new_reference_point,target box])

	:param im:
	:param bboxes:
	:param scores:
	:param reference_point:
	:param thr:
	:param visualization:
	:param margin: if the distance between reference point and target box is bigger than margin ,return None.
	:param IoUthreshold: if the IoU of the target box is bigger than IoUthreshold  ,return None.
	:return:
	"""
	indexs = np.where(scores >= thr)[0]
	if len(indexs) != 0 :
		thick = 4 # line thick
		imgcv = np.copy(im)
		new_bbox_det = []
		new_bbox_det = np.array(new_bbox_det)
		colors = (125
		          ,0,125)
		#  can change to parallel operation
		# for index in indexs[0]:
		#     box = bboxes[index]
		#     bottom_center = np.array((box[0], box[1], box[2], box[3], scores[index], int(0.5 * (box[0] + box[2])), box[3]))
		#     new_bbox_det = np.insert(new_bbox_det, 0, values=bottom_center, axis=0)
		if visualization == True:
			cv2.circle(imgcv,reference_point,5,(0,0,255),-1)
			for index in indexs:
				box = bboxes[index]
				cv2.rectangle(imgcv,
							  (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
							  colors, thick)
				mess = '%.3f' % (scores[index])
				cv2.putText(imgcv, mess, (int(box[0]), int(box[1] - 7)), 0, 0.5 , (0, 0 ,0 ) , 1)

		'''find the box that is nearlest to the img_point'''
		length = indexs.shape[0]
		#[xl,yl,xr,yr, bottom_center_x,bottom_center_y, scores] 7 items  hori
		new_bbox_det = np.hstack((bboxes[indexs],
								  np.expand_dims(0.5*(bboxes[indexs,0]+bboxes[indexs,2]),axis=1),
								  np.expand_dims(bboxes[indexs,3],axis=1),
								  np.expand_dims(scores[indexs],axis=1)))
		
		residual = new_bbox_det[:,4:6] - np.array([reference_point])
		residual = np.square(residual)
		res_sum = np.sqrt(np.sum(residual,axis=1))  #calculate the Standard deviation
		res_min = np.min(res_sum)
		
		index = int(np.where(res_sum==res_min)[0])
		'''target is the nearlest box to the img_point '''
		target = new_bbox_det[index,:]

		'''set margin to constraint the distance between reference point and target box.'''
		target_width = round(target[2]-target[0])
		# if res_min > min(margin, target_width):
		if res_min > target_width:
			# print('res_min = {:.2f} bigger  target_box_width = {:.2f}'.format(res_min, target_width))
			return (imgcv, None, [],[])
		# else:
		# 	print('res_min = {:.2f} smaller target_box_width = {:.2f}'.format(res_min, target_width))
		
		'''calculate the IoUs between target box and the other boxes.'''
		if length > 1:
			new_bbox_det = np.delete(new_bbox_det,index,axis=0)
			ixmin = np.maximum(new_bbox_det[:,0],target[0])
			iymin = np.maximum(new_bbox_det[:,1],target[1])
			ixmax = np.minimum(new_bbox_det[:,2],target[2])
			iymax = np.minimum(new_bbox_det[:,3],target[3])
			iw = np.maximum(ixmax - ixmin + 1., 0.)
			ih = np.maximum(iymax - iymin + 1., 0.)
			inters = iw * ih
			
			#union
			uni = ((target[2]-target[0] +1) * (target[3] - target[1] +1)
					+ (new_bbox_det[:,2]-new_bbox_det[:,0] + 1)*(new_bbox_det[:,3]-new_bbox_det[:,1] + 1)
					- inters)
			
			overlaps = inters / uni
			ovmax = np.max(overlaps)
			if ovmax > IoUthreshold:
				return (imgcv,None,[],[])
			
		box = target.astype(np.dtype('int64')).tolist()
		# The rectangular img of the target person
		sub_img = copy.copy(im[box[1]:box[3],box[0]:box[2]])
		# The new reference point is the bottom center of the target bbox .
		new_reference_point = (int(0.5*(box[0]+box[2])),box[3])

		'''calculate the distance between new and old reference point'''
		if visualization == True:
			cv2.circle(imgcv,new_reference_point,5,(0,255,255),-1)
			cv2.rectangle(imgcv,
						  (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
						  (0, 255, 255), thick)
			mess = 'res_min = {:.2f}, target_box_width = {:.2f}'.format(res_min,target_width)
			cv2.putText(imgcv, mess, (200,100), 0, 0.5, (0, 255, 255), 1)
		return (imgcv,sub_img,new_reference_point,[box[0],box[1],box[2]-box[0],box[3]-box[1]])
	else :
		return ([],None,[],[])

def Caculate_num(im, bboxes, CLASS_SET, thr=0.2, visualization = False,):

	'''If there is no bbox whose score is bigger than threshold,then return None'''
	indexs = np.where(bboxes[:,-2] >= thr)

	if len(indexs[0]) != 0 :

		imgcv = np.copy(im)
		new_bboxs_det = np.sort(bboxes[indexs,:],axis=0)[0] # boxes sorted by x_l.
		num = 0
		for num_count in range(len(new_bboxs_det)):
			label = int(new_bboxs_det[num_count,-1])
			num = 10*num + int(CLASS_SET[label])

		if visualization == True:
			thick = 1  # line thick
			colors = (125,0,125)
			for index in indexs[0]:
				box = bboxes[index]
				cv2.rectangle(imgcv,
							  (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
							  colors, thick)
				label = 'L:{} '.format(int(CLASS_SET[int(box[-1])]))
				score = 'S:{:.3f} '.format(box[-2])
				cv2.putText(imgcv, label, (int(box[0]), int(box[1]+2)), 0, 0.5 , (0,0,255) , 1)
				# cv2.putText(imgcv, score, (int(box[0]), int(box[3]-2)), 0, 0.5 , (0,0,255) , 1)
		return imgcv,num

	else :

		return (im,-1)

def show_detection(im, bboxes, CLASS_SET, thr=0.2, visualization = True):

	'''If there is no bbox whose score is bigger than threshold,then return None'''
	base = int(np.ceil(pow(len(CLASS_SET), 1. / 3)))
	colors = [_to_color(x, base) for x in range(len(CLASS_SET))]
	indexs = np.where(bboxes[:,-2] >= thr)
	if len(indexs[0]) != 0 :
		imgcv = np.copy(im)
		new_bboxs_det = np.sort(bboxes[indexs,:],axis=0)[0] # boxes sorted by x_l.

		if visualization == True:
			thick = 1  # line thick
			for index in indexs[0]:
				box = bboxes[index]
				cv2.rectangle(imgcv,
							  (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
							  colors[int(box[-1])], thick)
				label = 'L:{} '.format((CLASS_SET[int(box[-1])]))
				score = 'S:{:.3f} '.format(box[-2])
				cv2.putText(imgcv, label, (int(box[0]), int(box[1]+2)), 0, 0.5 , colors[int(box[-1])] , 1)
				cv2.putText(imgcv, score, (int(box[0]), int(box[3]-2)), 0, 0.5 , colors[int(box[-1])] , 1)
		return imgcv
	else :
		return im

def show_svhn(im, bboxes,CLASSES):

	'''If there is no bbox whose score is bigger than threshold,then return None'''
	length = len(bboxes)
	if length != 0 :
		thick = 1
		for index in range(length):

			box = bboxes[index ,:]
			colors = (125,0,125)
			cv2.rectangle(im,
						  (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
						  colors, thick)
			mess = '{}'.format(int(CLASSES[int(box[-1])]))
			cv2.putText(im, mess, (int(box[0]), int(box[1]+7)), 0, 0.5 , (0, 0 ,255 ) , 1)

		return im
	else :

		return im

def ScreenSHot(img_point,action_time, video_parameter, setting_parameter):
	'''每次使用这个函数的时候都需要重新调整时间。'''
	video = video_parameter['video']
	width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
	Message = GenerateRect(img_point,setting_parameter['Output_size'],setting_parameter['bias'],width,height)
	if Message[0] == True:

		time = action_time + video_parameter['delta_t'] #action time need to add the delta time to calibrate the time between channels .
		video.set(cv2.CAP_PROP_POS_MSEC, round(1000 * time))
		_, img = video.read()
		# print('--------------------------------------------------------')
		# print('currrent frames = ',video.get(cv2.CAP_PROP_POS_FRAMES))
		# print('currrent times = ', time)

		rect = Message[1]
		x_l = int(rect[0])
		y_l = int(rect[1])
		x_r = int(rect[2] + rect[0])
		y_r = int(rect[3] + rect[1])
		if type(img) != np.ndarray:
			return (False, None, None)

		sub_img = img[y_l:y_r,x_l:x_r]
		new_point = (int(img_point[0]-x_l),int(img_point[1] - y_l))
		return (True,sub_img, new_point, (x_l,y_l))
	else:
		return (False,None,None)


def ScreenSHot_batch(img_point, img, setting_parameter, width, height):
	'''
	与上面那个函数 ScreenSHot() 相比，使用这个函数，不需要每次都调整视频的时间，只需要在批量使用之前调整一次就行。
	'''
	Message = GenerateRect(img_point,setting_parameter['Output_size'],setting_parameter['bias'],width,height)
	if Message[0] == True:
		# print('--------------------------------------------------------')
		# print('currrent frames = ',video.get(cv2.CAP_PROP_POS_FRAMES))
		# print('currrent times = ', time)

		rect = Message[1]
		x_l = int(rect[0])
		y_l = int(rect[1])
		x_r = int(rect[2] + rect[0])
		y_r = int(rect[3] + rect[1])
		if type(img) != np.ndarray:
			return (False, None, None)

		sub_img = img[y_l:y_r,x_l:x_r]
		new_point = (int(img_point[0]-x_l),int(img_point[1] - y_l))
		return (True,sub_img, new_point, (x_l,y_l))
	else:
		return (False,None,None,None)

