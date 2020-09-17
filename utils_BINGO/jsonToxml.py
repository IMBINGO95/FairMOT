import json
import cv2
import sys
from scipy.io import loadmat
import h5py
import numpy as np
import shutil
import hdf5storage
import random
from collections import defaultdict

if sys.version_info[0] == 2:
		import xml.etree.cElementTree as ET
else:
		import xml.etree.ElementTree as ET
import os

def SVHN_json_to_xml(root, dataset):
		filename = os.path.join(root, dataset, dataset + '.json')
		with open(filename , 'r') as f:
				raw = json.load(f)

		for item in raw:
				img = cv2.imread(os.path.join(root, dataset, 'JPEGImages', item['name']))
				shape = img.shape
				xml_name = item['name'].split('.')[0] + '.xml'

				anno = ET.Element("annotation")
				ET.SubElement(anno, "folder").text = dataset
				ET.SubElement(anno, "filename").text = item["name"]

				source = ET.SubElement(anno, "source")
				ET.SubElement(source,"database").text = 'Unknown'

				size = ET.SubElement(anno, "size")
				ET.SubElement(size,"height").text = str(shape[0])
				ET.SubElement(size,"width").text = str(shape[1])
				ET.SubElement(size,"depth").text = str(shape[2])

				ET.SubElement(anno, "segmented").text = "0"
				bboxes = item["bbox"]
				for bbox in bboxes:
						ob = ET.SubElement(anno, "object")
						if bbox[4] == 10:
								bbox[4] == 0
						ET.SubElement(ob,"name").text = str(bbox[4])
						ET.SubElement(ob,"pose").text = "Unspecified"
						ET.SubElement(ob,"truncated").text = "0"
						ET.SubElement(ob,"difficult").text = "0"
						bndbox = ET.SubElement(ob,"bndbox")
						ET.SubElement(bndbox,"xmin").text = str(bbox[0])
						ET.SubElement(bndbox,"ymin").text = str(bbox[1])
						ET.SubElement(bndbox,"xmax").text = str(bbox[0] + bbox[2])
						ET.SubElement(bndbox,"ymax").text = str(bbox[1] + bbox[3])
				tree = ET.ElementTree(anno)
				tree.write('filename.xml')
				save = os.path.join(root,dataset,'Annotations',xml_name)
				tree.write(save)
				print(xml_name)

def SJN_json_to_xml(root, dataset):
		filename = os.path.join(root, dataset, dataset + '.json')
		with open(filename , 'r') as f:
				raw = json.load(f)

		for item in raw:
				img = cv2.imread(os.path.join(root, dataset, 'JPEGImages', item[0]))
				shape = img.shape
				xml_name = item[0].split('.')[0] + '.xml'

				anno = ET.Element("annotation")
				ET.SubElement(anno, "folder").text = dataset
				ET.SubElement(anno, "filename").text = item[0]

				source = ET.SubElement(anno, "source")
				ET.SubElement(source,"database").text = 'Unknown'

				size = ET.SubElement(anno, "size")
				ET.SubElement(size,"height").text = str(shape[0])
				ET.SubElement(size,"width").text = str(shape[1])
				ET.SubElement(size,"depth").text = str(shape[2])

				ET.SubElement(anno, "segmented").text = "0"

				if item[2] == 10:
						ob = ET.SubElement(anno, "object")
						ET.SubElement(ob, "name").text = str(int(item[1]))
						ET.SubElement(ob, "pose").text = "Unspecified"
						ET.SubElement(ob, "truncated").text = "0"
						ET.SubElement(ob, "difficult").text = "0"
						bndbox = ET.SubElement(ob, "bndbox")
						ET.SubElement(bndbox, "xmin").text = "0"
						ET.SubElement(bndbox, "ymin").text = "0"
						ET.SubElement(bndbox, "xmax").text = "0"
						ET.SubElement(bndbox, "ymax").text = "0"
						print('num = {}'.format(item[1]))
				else:
						ob = ET.SubElement(anno, "object")
						ET.SubElement(ob, "name").text = str(int(item[1]))
						ET.SubElement(ob, "pose").text = "Unspecified"
						ET.SubElement(ob, "truncated").text = "0"
						ET.SubElement(ob, "difficult").text = "0"
						bndbox = ET.SubElement(ob, "bndbox")
						ET.SubElement(bndbox, "xmin").text = "0"
						ET.SubElement(bndbox, "ymin").text = "0"
						ET.SubElement(bndbox, "xmax").text = "0"
						ET.SubElement(bndbox, "ymax").text = "0"

						ob1 = ET.SubElement(anno, "object")
						ET.SubElement(ob1, "name").text = str(int(item[2]))
						ET.SubElement(ob1, "pose").text = "Unspecified"
						ET.SubElement(ob1, "truncated").text = "0"
						ET.SubElement(ob1, "difficult").text = "0"
						bndbox1 = ET.SubElement(ob1, "bndbox")
						ET.SubElement(bndbox1, "xmin").text = "2"
						ET.SubElement(bndbox1, "ymin").text = "0"
						ET.SubElement(bndbox1, "xmax").text = "0"
						ET.SubElement(bndbox1, "ymax").text = "0"

						print('num = {}'.format(10*item[1] + item[2]))


				tree = ET.ElementTree(anno)
				tree.write('filename.xml')
				save = os.path.join(root,dataset,'Annotations',xml_name)
				tree.write(save)
				print(xml_name)

def SJN_json_to_NumberRegion_xml(root, dataset):
		filename = os.path.join(root, dataset, dataset + '.json')
		with open(filename , 'r') as f:
				raw = json.load(f)

		w_index = [3,5,7,9]
		h_index = [4,6,8,10]

		target_root = os.path.join(root, dataset, 'JPEGImages_NR')
		if not os.path.isdir(target_root):
				os.makedirs(target_root)
		anno_root = os.path.join(root, dataset, 'Annotations_NR')
		if not os.path.isdir(anno_root):
				os.makedirs(anno_root)

		for item in raw:
				img_path = os.path.join(root, dataset, 'JPEGImages', item[0])
				if len(item) == 3 or not os.path.isfile(img_path):
						continue
				else:

						if not os.path.isfile(os.path.join(target_root,item[0])):
								shutil.copyfile(img_path, os.path.join(target_root,item[0]))
						img = cv2.imread(img_path)
						height,weight,depth = img.shape
						xml_name = item[0].split('.')[0] + '.xml'

						w_candidate = [ round(item[i] * weight) for i in w_index ]
						h_candidate = [ round(item[j] * height) for j in h_index ]


						anno = ET.Element("annotation")
						ET.SubElement(anno, "folder").text = dataset
						ET.SubElement(anno, "filename").text = item[0]

						source = ET.SubElement(anno, "source")
						ET.SubElement(source,"database").text = 'Unknown'

						size = ET.SubElement(anno, "size")
						ET.SubElement(size,"height").text = str(weight)
						ET.SubElement(size,"width").text = str(height)
						ET.SubElement(size,"depth").text = str(depth)
						ET.SubElement(anno, "segmented").text = "0"

						ob = ET.SubElement(anno, "object")
						ET.SubElement(ob, "name").text = "region"
						ET.SubElement(ob, "pose").text = "Unspecified"
						ET.SubElement(ob, "truncated").text = "0"
						ET.SubElement(ob, "difficult").text = "0"

						if item[2] == 10:
								length = 1
								num = int(item[1])
						else:
								length = 2
								num = int(10 * item[1] + item[2])
						print('num = {}'.format(num))
						ET.SubElement(ob, "length").text = str(length)
						ET.SubElement(ob, "number").text = str(num)

						bndbox = ET.SubElement(ob, "bndbox")
						ET.SubElement(bndbox, "xmin").text = str(min(w_candidate))
						ET.SubElement(bndbox, "xmin").text = str(min(w_candidate))
						ET.SubElement(bndbox, "ymin").text = str(min(h_candidate))
						ET.SubElement(bndbox, "xmax").text = str(max(w_candidate))
						ET.SubElement(bndbox, "ymax").text = str(max(h_candidate))

						tree = ET.ElementTree(anno)
						tree.write('filename.xml')
						save = os.path.join(anno_root,xml_name)
						tree.write(save)
						print(xml_name)

						# cv2.rectangle(img,(min(w_candidate),min(h_candidate)),(max(w_candidate),max(h_candidate)),(0,0,255),1)
						# cv2.imwrite(os.path.join(target_root,item[0]),img)

def original_to_json(root,dataset):
		fileread = os.path.join(root, 'original_data', dataset + '.json')
		with open(fileread , 'r') as f:
				raw = json.load(f)

		raw = raw['digitStruct']
		data = []
		for element in raw:
				sub_data = {}
				sub_data['name'] = element['name']
				bboxes = element['bbox']
				if isinstance(bboxes, dict):
						x_l = bboxes['left']
						y_l = bboxes['top']
						w = bboxes['width']
						h = bboxes['height']
						if bboxes['label'] == 10:
								bboxes['label'] = 0
						label = bboxes['label']
						sub_data['bbox'] = [[x_l,y_l,w,h,label]]
						data.append(sub_data)
				elif isinstance(bboxes,list):
						target = []
						for box in bboxes:
								x_l = box['left']
								y_l = box['top']
								w = box['width']
								h = box['height']
								if box['label'] == 10:
										box['label'] = 0
								label = box['label']
								target.append([x_l, y_l, w, h, label])
						sub_data['bbox'] = target
						print(len(target))
						data.append(sub_data)


		filewrite = os.path.join(root, dataset, dataset + '.json')
		with open(filewrite , 'w') as f:
				json.dump(data,f)
		print()

def mat_to_json(root,dataset):
		filename = os.path.join(root, dataset, 'digitStruct.mat')
		data = loadmat(filename,struct_as_record=False, squeeze_me=True)
		f = h5py.File(filename)
		print(list(f.keys()))
		x = f['digitStruct']
		print(list(x.keys()))


		y = np.array(x)
		print()

def SJN_JSON_with_region(root,dataset):

		filename = os.path.join(root, dataset, dataset + '.json')
		with open(filename, 'r') as f:
				raw = json.load(f)

		w_index = [3, 5, 7, 9]
		h_index = [4, 6, 8, 10]

		# target_root = os.path.join(root, dataset, 'JPEGImages')
		# if not os.path.isdir(target_root):
		#     os.makedirs(target_root)
		# anno_root = os.path.join(root, dataset, 'Annotations_NR')
		# if not os.path.isdir(anno_root):
		#     os.makedirs(anno_root)

		target = []
		for item in raw:
				if item[2] == 10:
						length = 1
						num = int(item[1])
				else:
						length = 2
						num = int(10 * item[1] + item[2])

				if len(item) == 3 :
						target.append([item[0], 0,0,0,0, int(item[1]),int(item[2]),length])

				else:
						img_path = os.path.join(root, dataset, 'JPEGImages', item[0])
						img = cv2.imread(img_path)
						height, weight, depth = img.shape



						w_candidate = [round(item[i] * weight) for i in w_index]
						h_candidate = [round(item[j] * height) for j in h_index]

						target.append([item[0], min(w_candidate),min(h_candidate),max(w_candidate),max(h_candidate),int(item[1]),int(item[2]),length])

		filename = os.path.join(root, dataset, dataset + 'Region_all.json')
		with open(filename, 'w') as f:
				json.dump(target,f)

def Generate_neg_json(root,dataset,img_dir):
		img_candidate = os.listdir(os.path.join(root,dataset,img_dir))
		target = []
		for img_name in img_candidate:
				target.append([img_name,0,0,0,0,10,10,0])
		with open(os.path.join(root,dataset,img_dir+'.json'),'w') as f:
				json.dump(target,f)

def move_imgs(root,dataset,img_dir,json_file):
		with open(os.path.join(root,dataset,json_file),'r') as f:
				data = json.load(f)

		for index,item in enumerate(data):
				shutil.copyfile(os.path.join(root,dataset,img_dir,item[0]),os.path.join(root,dataset,'JPEGImagesRegion',item[0]))

		return data


def Channel1_json_to_xml(root,img_root,json_file):
	imgs_name = [img for img in os.listdir(img_root) if img[-3:] == 'jpg' and int(img[:-4]) < 2500]
	
	with open(json_file, 'r') as f:
		origin = json.load(f)
	
	Frame_dict = defaultdict(list)
	Num_dict = defaultdict(list)
	
	for item in origin:
		Frame_dict[item[0]] += [item]
		Num_dict[item[-1]] += [item]
	imgs_name.sort()
	random.shuffle(imgs_name)
	
	train = imgs_name[:2000]
	test = imgs_name[2000:]
	
	for target_root,target in (('train',train),('test',test)):
		if not os.path.exists(os.path.join(root,target_root,'JPEGImages')):
			os.makedirs(os.path.join(root,target_root,'JPEGImages'))
			
		if not os.path.exists(os.path.join(root,target_root,'Annotations')):
			os.makedirs(os.path.join(root,target_root,'Annotations'))
			
		for img_name in target:
			img_int = int(img_name[:-4])
			shutil.copyfile(os.path.join(img_root,img_name),os.path.join(root,target_root,'JPEGImages',img_name))
			data = Frame_dict[img_int]
			
			xml_name = os.path.join(root,target_root,'Annotations',img_name[:-4]+'.xml')
			
			# imgcv = cv2.imread(os.path.join(img_root, img_name))
			# shape = imgcv.shape
			anno = ET.Element("annotation")
			ET.SubElement(anno, "folder").text = os.path.join(root, target_root, 'JPEGImages')
			ET.SubElement(anno, "filename").text = img_name
			
			source = ET.SubElement(anno, "source")
			ET.SubElement(source, "database").text = 'Unknown'
			
			size = ET.SubElement(anno, "size")
			# ET.SubElement(size, "height").text = str(shape[0])
			# ET.SubElement(size, "width").text = str(shape[1])
			# ET.SubElement(size, "depth").text = str(shape[2])
			ET.SubElement(size, "height").text = '1450'
			ET.SubElement(size, "width").text = '5950'
			ET.SubElement(size, "depth").text = '3'
			
			ET.SubElement(anno, "segmented").text = "0"
			
			for item in data:
				ob = ET.SubElement(anno, "object")
				ET.SubElement(ob, "name").text = 'person'
				ET.SubElement(ob, "pose").text = "Unspecified"
				ET.SubElement(ob, "truncated").text = "0"
				ET.SubElement(ob, "difficult").text = "0"
				bndbox = ET.SubElement(ob, "bndbox")
				ET.SubElement(bndbox, "xmin").text = str(item[2])
				ET.SubElement(bndbox, "ymin").text = str(item[3])
				ET.SubElement(bndbox, "xmax").text = str(item[2] + item[4])
				ET.SubElement(bndbox, "ymax").text = str(item[3] + item[5])
				
			# 	cv2.rectangle(imgcv,
			# 	              (int(item[2]), int(item[3])), (int(item[2]+item[4]), int(item[3]+item[5])),
			# 	              (0,0,255), 1)
			# cv2.imwrite(os.path.join(root,'show',img_name),imgcv)
				
			tree = ET.ElementTree(anno)
			tree.write('filename.xml')
			tree.write(xml_name)
			print(xml_name)


if __name__ == '__main__':
	Channel1_json_to_xml('/datanew/hwb/data/Football/ChanOne/','/datanew/hwb/data/Football/ChanOne/origin_img','/datanew/hwb/data/Football/ChanOne/2500FrameGT.json')

