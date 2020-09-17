import json
import cv2
import os
import collections
from xml.dom.minidom import Document

def creatChild(main_doc,parent_item,item_name,item_text):
    '''

    :param main_doc:  This is a Document() item.
    :param parent_item:
    :param item:
    :param item_text:
    :return:
    '''
    item = main_doc.createElement(item_name)
    text = main_doc.createTextNode(item_text)
    item.appendChild(text)
    parent_item.appendChild(item)
    return parent_item


def gtsToxml(dataset_name = 'VOC2019', file_dir = '',img_ID = None, size = (512, 512, 3), gts = []):
    '''
    wrtie gts to xml
    :param dataset_name:
    :param file_dir:
    :param img_ID:
    :param size:
    :param gts:
    :return:
    '''
    doc = Document()
    # root
    root = doc.createElement('annotation')
    doc.appendChild(root)
    # item 1 : folder
    creatChild(doc ,root, 'annotation', dataset_name)

    # item 2 :filename
    creatChild(doc, root, 'filename', img_ID + '.jpg')

    # item 3 :source
    item3 = doc.createElement('source')
    root.appendChild(item3)
    # item 3_1 : database
    creatChild(doc, item3, 'database', 'The '+ dataset_name + ' Database')
    # item 3_2 :annotation
    creatChild(doc, item3, 'annotation', 'PASCAL '+ dataset_name)
    # item 3_3 :image
    creatChild(doc, item3, 'image', 'AI_GROUP')
    # item 3_4 :AI_GROUP_ID
    creatChild(doc, item3, 'AI_GROUP_ID', img_ID)

    # item 4 : owner
    item4 = doc.createElement('owner')
    root.appendChild(item4)
    # item 4_1 :GROUP
    creatChild(doc, item4, 'GROUP', 'AI_GROUP')
    # item 4_2 :name
    creatChild(doc, item4, 'name', 'BINGO')

    # item 5 : size
    item5 = doc.createElement('size')
    root.appendChild(item5)
    # item 5_1 : width
    creatChild(doc, item5, 'width', '{}'.format(size[0]))
    # item 5_2 : heigth
    creatChild(doc, item5, 'heigth', '{}'.format(size[1]))
    # item 5_3 : depth
    creatChild(doc, item5, 'depth', '{}'.format(size[2]))

    # item 6 : segmented
    # 0 means no segmented , 1 means segmented
    creatChild(doc, root, 'segmented', '{}'.format(1))

    # Object item
    for obeject in gts:
        obeject_item = doc.createElement('object')
        root.appendChild(obeject_item)
        # item obeject_1 : name
        creatChild(doc, obeject_item, 'name', '{}'.format(obeject[0]))
        # item obeject_2 : pose
        creatChild(doc, obeject_item, 'pose', '{}'.format(obeject[1]))
        # item obeject_3 : truncated
        creatChild(doc, obeject_item, 'truncated', '{}'.format(obeject[2]))
        # item obeject_4 : difficult
        creatChild(doc, obeject_item, 'difficult', '{}'.format(obeject[3]))

        # item obeject_5 : bndbox
        obeject_item_5 = doc.createElement('bndbox')
        obeject_item.appendChild(obeject_item_5)
        # item obeject_5_1 : xmin
        creatChild(doc, obeject_item_5, 'xmin', '{}'.format(obeject[4]))
        # item obeject_5_2 : ymin
        creatChild(doc, obeject_item_5, 'ymin', '{}'.format(obeject[5]))
        # item obeject_5_3 : xmax
        creatChild(doc, obeject_item_5, 'xmax', '{}'.format(obeject[6]))
        # item obeject_5_4 : ymax
        creatChild(doc, obeject_item_5, 'ymax', '{}'.format(obeject[7]))

    # 将DOM对象doc写入文件
    file_path = os.path.join(file_dir,img_ID+'.xml')
    f = open(file_path, 'w')
    # f.write(doc.toprettyxml(indent = '\t', newl = '\n', encoding = 'utf-8'))
    doc.writexml_1(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8', tagName=root.tagName)

    f.close()
