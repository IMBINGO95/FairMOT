import os
import codecs
import json


if __name__ == '__main__':

    root = '/datanew/hwb/data/Football/SoftWare/'
    eval_index = '10'
    root_path = os.path.join(root, '{}'.format(eval_index))

    before_label_file = '{}.json'.format(eval_index)
    rectify_label_file = 'rectified_{}'.format(eval_index)

    if os.path.exists(os.path.join(root_path, rectify_label_file)):
        true_label_file = rectify_label_file
    # 0
    rectify_items = {
        31:6,
        53:23,

    }

    # load in true label information
    with codecs.open(os.path.join(root_path, before_label_file), 'r','utf-8-sig') as f:
        original_data = json.load(f)

    action_data = original_data['data']

    for item_index in rectify_items.keys():
        old_num = action_data[item_index]['num']
        new_num = rectify_items[item_index]
        action_data[item_index]['num'] = '{}'.format(rectify_items[item_index])
        print('{}\'s old num = {}, rectified num = {}'.format(item_index,old_num,new_num))


    original_data['data'] = action_data

    with codecs.open(os.path.join(root_path, rectify_label_file), 'w','utf-8-sig') as f:
        json.dump(original_data,f)

