from termcolor import cprint
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
from M2Det.utils.forward import NumReg_detect
import numpy as np
import cv2
import os
from SPPE.dataloader import SPPE_img_preprocess
from SPPE.src.utils.eval import getPrediction_batch

def print_info(info, _type=None):
	if _type is not None:
		if isinstance(info,str):
			cprint(info, _type[0], attrs=[_type[1]])
		elif isinstance(info,list):
			for i in range(info):
				cprint(i, _type[0], attrs=[_type[1]])
	else:
		print(info)

def _loss(length_logits, digits_logits, length_labels, digits_labels):
	'''calculate loss of correct number of this batchsize'''
	length_predictions = length_logits.data.max(1)[1]
	digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]

	needs_include_length = True
	num_correct = 0
	if needs_include_length:
		num_correct += (length_predictions.eq(length_labels.data) &
						digits_predictions[0].eq(digits_labels[0].data) &
						digits_predictions[1].eq(digits_labels[1].data)).cpu().sum()
	else:
		num_correct += (digits_predictions[0].eq(digits_labels[0].data) &
						digits_predictions[1].eq(digits_labels[1].data)).cpu().sum()

	length_cross_entropy = torch.nn.functional.cross_entropy(length_logits, length_labels)
	digit1_cross_entropy = torch.nn.functional.cross_entropy(digits_logits[0], digits_labels[0])
	digit2_cross_entropy = torch.nn.functional.cross_entropy(digits_logits[1], digits_labels[1])

	loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy
	return (loss, num_correct)

def _adjust_learning_rate(optimizer, step, initial_lr, decay_steps, decay_rate):
	lr = initial_lr * (decay_rate ** (step // decay_steps))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr


def set_optimizer(net, cfg):
	return torch.optim.SGD(net.parameters(),
					 lr = cfg.train_cfg.learning_rate,
					 momentum = cfg.optimizer.momentum,
					 weight_decay = cfg.optimizer.weight_decay)

def img_forward(SVHN, img, mode, scores=False):

	if mode == 'test':
		transform = transforms.Compose([
			transforms.Resize([54,54]),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])
	else:
		transform = transforms.Compose([
			transforms.Resize([64,64]),
			transforms.RandomCrop([54, 54]),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])
	# IMG = image
	# print(img.shape)
	image = Image.fromarray(img)
	# print_info('img.size = {}, image.size = {}'.format(img.shape,image.size))
	image = transform(image).unsqueeze(0)
	images = Variable(image.cuda(), volatile=True)
	length_logits, digits_logits = SVHN(images)
	'''This max function return two column, the first row is value, and the second row is index '''
	length_predictions = length_logits.data.max(1)[1]
	length_score = length_logits.data.max(1)[0].cpu()
	digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]
	digits_score = [digit_logits.data.max(1)[0].cpu() for digit_logits in digits_logits]


	return int(length_predictions), [int(digits_predictions[0]),int(digits_predictions[1])],\
		   [float(length_score),float(digits_score[0]),float(digits_score[1])]

def Predict_Number(imgs,SvhnNet,NumReg_preprocess,NumReg_DetectNet,NumReg_detector,NumReg_priors,NumRegDetCfg,save_root=None):
	'''
	given a img array that contain many imgs.
	detect the number region of each img,and predict the number of this region.
	finnally return the number that maximumlly counted.
	:param imgs:
	:param SvhnNet:
	:param NumReg_preprocess:
	:param NumReg_DetectNet:
	:param NumReg_detector:
	:param NumReg_priors:
	:param NumRegDetCfg:
	:param save_root: If to visualize the result.
	:return:
	'''
	
	'''If to visualize the results of the process!'''
	if save_root != None:
		save_path = os.path.join(save_root, 'NumReg')
		visualization = True
		if not os.path.exists(save_path):
			os.makedirs(save_path)
	else:
		visualization = False
		
	
	NumsArray = []
	
	process_data = {'length_pres':[],'digits_pres':[],'nums_score':[]}
	
	for index,sub_img in enumerate(imgs):
		
		if sub_img.size <=0 :
			continue
		#####################
		#Numbe Region Detect#
		#####################
		RegCandidate,Reg_score = NumReg_detect(sub_img,
		                             NumReg_preprocess,
		                             NumReg_DetectNet,
		                             NumReg_detector,
		                             NumReg_priors,
		                             NumRegDetCfg,
                                     visualization)
		
		# # if RegCandidate.size() != torch.Size([]):
		# if save_root != None:
		# 	if type(RegCandidate) == np.ndarray:
		# 		cv2.imwrite(os.path.join(save_path,'{}.jpg'.format(index)), sub_img)
		
		if isinstance(RegCandidate, np.ndarray) and RegCandidate.size >0 :
			#################
			#Number Classify#
			#################
			length_pre, digits_pre, num_score = img_forward(SvhnNet, RegCandidate, 'test')
			process_data['length_pres'].append(length_pre)
			process_data['digits_pres'].append(digits_pre)
			process_data['nums_score'].append(num_score)

			if length_pre == 1:
				Num_pre = digits_pre[0]
			elif length_pre == 2:
				Num_pre = 10 * digits_pre[0] + digits_pre[1]
			else:
				continue
			NumsArray.append(Num_pre)
			
			if save_root != None:
				cv2.imwrite(os.path.join(save_path, '{}_{}_{}_{:.3f}.jpg'.format(index, length_pre, Num_pre, Reg_score)), sub_img)
		else:
			continue
			
	if len(NumsArray) > 0:
		# NumberArray range from 0 to 99.
		# We need to count how many times does each number appear!
		NumsArray = np.histogram(NumsArray, bins=100, range=(0, 100))[0]
		preNum = np.argmax(NumsArray)
		if preNum == 10:
			print('wrong value')
		preNum_count = NumsArray[preNum]
		if np.where(NumsArray == preNum_count)[0].size > 1:
			# if there are more than one number have the maximun counts, then return -1
			# can sort by number classification scores.
			preNum = -1
	else:
		preNum = -1
		
	return preNum,process_data


def Predict_Number_based_PoseEstimate(imgs, SvhnNet, pose_model, PDNet_based_on_hms, inputResH, inputResW, outputResH, outputResW, save_root=None):
	'''
	given a img array that contain many imgs.
	detect the number region of each img,and predict the number of this region.
	finnally return the number that maximumlly counted.
	'''

	'''If to visualize the results of the process!'''
	if save_root != None:
		save_path = os.path.join(save_root, 'NumReg')
		visualization = True
		if not os.path.exists(save_path):
			os.makedirs(save_path)
	else:
		visualization = False

	Number_before_proccess = len(imgs)
	# Img Preprocess
	inps = []
	uls = []
	brs = []
	for img in imgs:
		height, width, _ = img.shape
		if height==0 or width==0:
			continue
		upper_left = torch.from_numpy(np.array([0, 0]))
		bottom_right = torch.from_numpy(np.array([width, height]))
		inp = SPPE_img_preprocess(img)

		inps.append(inp)
		uls.append(upper_left)
		brs.append(bottom_right)
	# from numpy to torch
	inps = torch.stack(inps)
	uls = torch.stack(uls)
	brs = torch.stack(brs)

	# Pose estimation
	hms = pose_model(inps.cuda()).cpu()

	# 从heat_maps 中还原出各个关节点的坐标。
	preds_img, preds_scores = getPrediction_batch(hms, uls.float(), brs.float(), inputResH, inputResW,outputResH, outputResW)

	# 判断左胯右胯,左肩右肩的相对相对距离, 进行第一次筛选
	left_max = \
	torch.max(torch.cat((preds_img[:, 5, 0].unsqueeze(dim=1), preds_img[:, 11, 0].unsqueeze(dim=1)), dim=1), dim=1)[0]
	right_min = \
	torch.max(torch.cat((preds_img[:, 6, 0].unsqueeze(dim=1), preds_img[:, 12, 0].unsqueeze(dim=1)), dim=1), dim=1)[0]

	Alter_index_first = torch.gt(right_min, left_max).nonzero()
	if Alter_index_first.size(0) == 0:  # 都不满足要求, 号码返回-1，
		return -1
	Alter_index_first = Alter_index_first.squeeze(dim=1)

	# 通过关键节点判断是否是正确朝向。
	hms_sort_first = hms[Alter_index_first, 5:, :, :]
	#把多张Heat_maps拼成一张Heat_map
	input_size = hms_sort_first.size(1)
	Combined_input = hms_sort_first[:,0,:,:]
	for i in range(1, input_size):
		Combined_input += hms_sort_first[:,i,:,:]

	Combined_input = Combined_input.unsqueeze(dim=1)

	# print('Combined_input.size() : ', Combined_input.size() )
	# print_info(hms_sort_first)
	Direction_preds = PDNet_based_on_hms(Combined_input.cuda())
	Preds = Direction_preds.max(dim=1)[1]
	# print('Preds.size {}'.format(Preds.size()))
	sort_array = Preds.nonzero()
	if sort_array.size(0) == 0:
		return -1
	# print('batch index = {} ,sort_array.size {}'.format(i,sort_array.size()))
	# print('batch index = {} ,sort_array.dim {}'.format(i,sort_array.dim()))
	sort_array = sort_array.squeeze(dim=1)
	Alter_index_second = Alter_index_first[sort_array]

	# 进行两次筛选之后，选择朝向符合的图片,并对图片进行预处理。
	imgs_SVHN, rectangle_imgs = img_pre_for_SVHN(imgs, preds_img, Alter_index_second.tolist(),visualization)

	if type(imgs_SVHN) != torch.Tensor:
		return -1

	# predict the number of the image.
	length_logits, digits_logits = SvhnNet(imgs_SVHN.cuda())
	'''This max function return two column, the first row is value, and the second row is index '''
	length_predictions = length_logits.max(1)[1].cpu().tolist()
	digits_predictions = [digit_logits.max(1)[1].cpu().tolist() for digit_logits in digits_logits]

	NumsArray = []
	for Num_i in range(len(length_predictions)):
		Number_len = length_predictions[Num_i]
		if Number_len == 1 :
			Num = digits_predictions[0][Num_i]
		elif Number_len == 2 :
			Num = digits_predictions[0][Num_i]*10 + digits_predictions[1][Num_i]
		else:
			Num = -1
		NumsArray.append(Num)
		cv2.imwrite(os.path.join(save_path, '{}_P{}.jpg'.format(Num_i,Num)), rectangle_imgs[Num_i])

	Number_after_proccess = len(NumsArray)

	print('Number_before_proccess : {}, Number_after_proccess : {} , delete : {} '.format(Number_before_proccess,Number_after_proccess,Number_before_proccess-Number_after_proccess ))

	if len(NumsArray) > 1:
		# NumberArray range from 0 to 99.
		# We need to count how many times does each number appear!
		NumsArray = np.histogram(NumsArray, bins=100, range=(0, 100))[0]
		preNum = np.argmax(NumsArray)
		if preNum == 10:
			print('wrong value')
		preNum_count = NumsArray[preNum]
		if np.where(NumsArray == preNum_count)[0].size > 1:
			# if there are more than one number have the maximun counts, then return -1
			# can sort by number classification scores.
			preNum = -1
	else:
		preNum = -1



	return preNum


def img_pre_for_SVHN(origin_imgs, preds_img, Alter_index,visualization=False):

	transform = transforms.Compose([
		transforms.Resize([54, 54]),
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])

	thick = 1  # line thick
	colors = (125, 0, 125)

	imgs_array = []
	rectangle_imgs = []
	for index in Alter_index:
		ul_x = round(min(preds_img[index, 5, 0].item(), preds_img[index, 11, 0].item()))
		ul_y = round(min(preds_img[index, 5, 1].item(), preds_img[index, 6, 1].item()))
		br_x = round(max(preds_img[index, 6, 0].item(), preds_img[index, 12, 0].item()))
		br_y = round(max(preds_img[index, 11, 1].item(), preds_img[index, 12, 1].item()))
		origin_img = origin_imgs[index]
		i_height, i_weight, i_channel = origin_img.shape
		if visualization == True:
			cv2.rectangle(origin_img, (ul_x, ul_y), (br_x, br_y), colors, thick)
		crop_img = origin_img[max(ul_y, 0):min(i_height, br_y), max(ul_x, 0):min(br_x, i_weight)]
		h_i, w_i, _ = crop_img.shape
		if h_i < 21 or w_i < 12:
			# 如果背部区域太小了，也要舍弃。
			continue
		rectangle_imgs.append(origin_img)
		image = Image.fromarray(crop_img)
		image = transform(image)
		imgs_array.append(image)

	if len(imgs_array) == 0:
		return None, None

	imgs_array = torch.stack(imgs_array, dim=0)
	return imgs_array, rectangle_imgs