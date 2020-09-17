import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
import json
import shutil
class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, CLASSES, keep_difficult=True):
        self.class_to_ind = dict(zip(CLASSES, range(len(CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))

        # difficult = int(obj.find('difficult').text) == 1
        # if not self.keep_difficult and difficult:
        #     continue
        size = target.find('size')
        height = size.find('height').text
        width = size.find('width').text
        depth = size.find('depth').text

        obj = target.find('object')
        name = obj.find('name').text.lower().strip()
        ## 返回数字
        # number = int(obj.find('number').text)
        # length = int(obj.find('length').text)
        ##返回字符串
        number = obj.find('number').text
        length = obj.find('length').text

        # bbox = obj.find('bndbox')
        # pts = ['xmin', 'ymin', 'xmax', 'ymax']
        # bndbox = []
        # for i, pt in enumerate(pts):
        #     cur_pt = int(bbox.find(pt).text)
        #     # scale height or width
        #     # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
        #     bndbox.append(cur_pt)
        # print(name)
        # label_idx = int(name)
        # 类是用来识别区的。
        label_idx = self.class_to_ind[name]
        # label_idx = int(name)
        # print(label_idx)
        # bndbox.append(label_idx)
        # bndbox.extend([length,number])
        # res = np.vstack((res, bndbox)).astype('int64')  # [xmin, ymin, xmax, ymax, label_ind,length,number]
        # img_id = target.find('filename').text[:-4]

        return width,height,depth,length,number

def read_xml(file_name,xml_transform):
    target = ET.parse(file_name).getroot()
    target = xml_transform(target)
    return target

def write_rectangle_dawei(root_dir,CLASSES,save_dir):
    xml_transform = AnnotationTransform(CLASSES)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    item_list = os.listdir(root_dir)
    for item in item_list:
        if item[-3:] != 'xml':
            continue
        xml_name = os.path.join(root_dir, item)
        sub_data = read_xml(xml_name,xml_transform)
        # new_name = xml_name[:-4]
        # os.rename(xml_name,new_name)
        img_name = item[:-8] + '.jpg'
        img = cv2.imread(os.path.join(root_dir,img_name))
        for line in sub_data:
            mess = '{}'.format(line[-1])
            cv2.rectangle(img,(line[0],line[1]),(line[2],line[3]),(0,0,255),1)
            cv2.putText(img,mess,(line[0],line[1]-7),1,1,(0,0,255),1)
        cv2.imwrite(os.path.join(save_dir,img_name),img)


def write_xml(save_dir,width,height,depth,name,length,num=None,item=None):

    anno = ET.Element("annotation")
    xml_name = '{}.xml'.format(name)
    ET.SubElement(anno, "folder").text = save_dir
    ET.SubElement(anno, "filename").text = '{}.jpg'.format(name)

    source = ET.SubElement(anno, "source")
    ET.SubElement(source, "database").text = 'Unknown'

    size = ET.SubElement(anno, "size")
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(anno, "segmented").text = "0"

    ob = ET.SubElement(anno, "object")
    ET.SubElement(ob, "name").text = "region"
    ET.SubElement(ob, "pose").text = "Unspecified"
    ET.SubElement(ob, "truncated").text = "0"
    ET.SubElement(ob, "difficult").text = "0"

    ET.SubElement(ob, "length").text = str(length)
    ET.SubElement(ob, "number").text = str(num)

    if item != None:
        bndbox = ET.SubElement(ob, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(round(item[0]))
        ET.SubElement(bndbox, "ymin").text = str(round(item[1]))
        ET.SubElement(bndbox, "xmax").text = str(round(item[2]))
        ET.SubElement(bndbox, "ymax").text = str(round(item[3]))

    tree = ET.ElementTree(anno)
    tree.write('filename.xml')
    save = os.path.join(save_dir, xml_name)
    tree.write(save)
    # print(xml_name)

def write_Posebased_region_number_recognition_xml(root_path,mode='train'):

    json_file = os.path.join(root_path,'PD_dataset',mode,'Annotations.json')
    save_dir = os.path.join(root_path,'PD_dataset',mode,'Annotations')
    os.makedirs(save_dir,exist_ok=True)

    with open(json_file,'r') as f :
        data = json.load(f)

    target_transform = AnnotationTransform(['region'])

    for item in data:
        if item['Label'] == 0:
            continue
        img_id = item['image_id'].split('.')[0]
        keypoints = item['keypoints']
        #四个关节点确定的区域。
        xmin = min(keypoints[5*3],keypoints[11*3])
        ymin = min(keypoints[5*3+1],keypoints[6*3+1])
        xmax = max(keypoints[6*3],keypoints[12*3])
        ymax = max(keypoints[11*3],keypoints[12*3+1])
        xml_read_path = os.path.join(root_path,mode,'Annotations','{}.xml'.format(img_id))
        width,height,depth,length,number = read_xml(xml_read_path,target_transform)
        write_xml(save_dir,width,height,depth,img_id,length,num=number,item=[xmin,ymin,xmax,ymax])
        print(item)

def write_SJK_210k_xml(root_path,mode='train'):

    json_file = os.path.join(root_path,mode,'{}.json'.format(mode))
    save_dir = os.path.join(root_path,mode,'Annotations_Pose')
    os.makedirs(save_dir,exist_ok=True)

    with open(json_file,'r') as f :
        data = json.load(f)

    target_transform = AnnotationTransform(['region'])

    for item in data:
        img_id = item[0].split('.')[0]
        if int(item[1]) == 10 and int(item[2]) == 10:
            length = 0
            number = None
        elif int(item[2]) == 10 :
            length = 1
            number = item[1]
        else:
            length = 2
            number = item[1] * 10 + item[2]
        img_read_path = os.path.join(root_path,mode,'JPEGImages','{}.jpg'.format(img_id))
        frame = cv2.imread(img_read_path)
        height, width, depth = frame.shape
        # width,height,depth = read_xml(xml_read_path,target_transform)
        write_xml(save_dir,width,height,depth,img_id,length,num=int(number),item=[0,0,0,0])
        print(item)




if __name__ == '__main__':
    # root_dir = r'G:\A_Dataset\Number_Detect\MaLaData\eval\Annotations'
    # save_dir = r'G:\A_Dataset\Number_Detect\MaLaData\eval_1'
    # CLASSES = ['background','region']
    # mode = 'test'
    # DigitRegion2NumberRegion(root_dir, CLASSES, save_dir, mode)
    root_path = '/datanew/hwb/data/SJN-210k/'
    write_SJK_210k_xml(root_path,mode='test')





