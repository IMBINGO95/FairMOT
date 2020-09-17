import json
import os
import codecs
import numpy as np
from utils_BINGO.Number_Rectifier import Number_Rectifier
from termcolor import cprint
import shutil

def print_info(info, _type=None):
	if _type is not None:
		if isinstance(info,str):
			cprint(info, _type[0], attrs=[_type[1]])
		elif isinstance(info,list):
			for i in range(info):
				cprint(i, _type[0], attrs=[_type[1]])
	else:
		print(info)


if __name__ == '__main__':

	root = '/datanew/hwb/data/Football/SoftWare/'
	eval_indexes = ['146-x']
	move_file = False
	Is_Number_Rectify = False

	# iterate root paths
	for eval_index in eval_indexes:

		root_path = os.path.join(root,'{}'.format(eval_index.split('-')[0]), '{}'.format(eval_index))

		true_label_file = '{}.json'.format(eval_index)

		print_info('=============================file path = {}==========================='.format(true_label_file), ['blue', 'bold'])

		for step in [1,2,3]:
			predict_label_file = 'Step{}_{}.json'.format(step, eval_index)
			# predict_label_file = 'with_VOT_test_1_after_sort.json'
			all_stacks = {}
			distmat_stacks = {}

			vis_dir = os.path.join(root_path, 'vis')
			# 错误的图片
			error_dir = os.path.join(root_path,'{}_eval'.format(root_path.split('/')[-1]),'error')
			miss_dir = os.path.join(root_path,'{}_eval'.format(root_path.split('/')[-1]),'miss')

			# 先删除，后生成
			if move_file == True:
				if os.path.exists(error_dir):
					shutil.rmtree(error_dir)
				os.makedirs(error_dir)
				#丢失的图片
				if os.path.exists(miss_dir):
					shutil.rmtree(miss_dir)
				os.makedirs(miss_dir)

			print_info('root_path = {}'.format(root_path),['yellow','bold'])

			# load in true label information
			with codecs.open(os.path.join(root_path, true_label_file), 'r','utf-8-sig') as f:
				true_labels = json.load(f)['data']
			# load in predited information
			if Is_Number_Rectify == True:
				Rectifier = Number_Rectifier(os.path.join(root_path, predict_label_file))
				predict_labels = Rectifier.rectify()
			else:
				with codecs.open(os.path.join(root_path, predict_label_file), 'r', 'utf-8-sig') as f:
					predict_labels = json.load(f)['data']


			total_box_num = 0
			total_num = len(predict_labels)
			predict_num = 0
			total_missed, num_pre_missed = 0,0
			correct_num ,wrong_num = 0,0
			wrong_index = []

			for i in range(total_num):
				t_label = true_labels[i]
				p_label = predict_labels[i]
				t_num = t_label['num']
				p_num = p_label['num']
				team = p_label['team']

				if team == None:
					total_missed += 1

				elif t_num == '门将' or t_num == None:
					correct_num += 1

				elif p_num == None or p_num=='':
					# not detection at all
					total_missed += 1
					predict_labels[i]['team'] = 'total_missed_' + str(predict_labels[i]['team'])

				else:
					if p_num == '-1':
						# the number did not be detected.
						num_pre_missed += 1
						predict_labels[i]['team'] = 'num_pre_missed_' + str(predict_labels[i]['team'])

					elif p_num == t_num or (p_num == '1' and t_num == '门将'):
						correct_num += 1

					else:
						wrong_num += 1
						predict_labels[i]['team'] = 'WW_{}_'.format(t_num) + str(predict_labels[i]['team'])
						wrong_index.append(i)
						if move_file == True:
							shutil.copytree(os.path.join(vis_dir,'{}'.format(i)),os.path.join(error_dir,'{}_T{}_P{}'.format(i,t_num,p_num)))
					# total_box_num += len(predict_labels[i]['process_data']['length_pres'])

			predict_num = wrong_num + correct_num
			stack = {}
			nums = len(predict_labels)
			hundred = 0

			while 100 * hundred < nums:
				stack[hundred] = predict_labels[hundred * 100:min(hundred * 100 + 100, nums)]
				hundred += 1
			all_stacks[root_path] = stack

			print('total num = {}, total missed num = {}'.format(total_num,total_missed))
			print('predict num = {}, predict_rate = {:.2f}'.format(predict_num,predict_num/total_num))
			print('correct num = {}, accuracy = {:.2f}'.format(correct_num,correct_num/predict_num))
			print(wrong_index)
		# print('per_action_boxes num = {:.2f}'.format(total_box_num/predict_num))
	print()