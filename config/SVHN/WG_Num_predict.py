
model = dict(
    input_size = 320,
    init_net = True,
    pretrained = 'None', # res series, directly load the pretrained weights when defining them
    rgb_means = (104, 117, 123),
	label_type = 'both' , #there are both positive label and negative label,use one of these or use both,
	crop_type = True # if to crop the image ,True what kinds of crop style
     )

train_cfg = dict(
	mode = 'train', #'directory to train data path'
    cuda = True,
    warmup = 5,
    batch_size = 128, # Default 64
	learning_rate = 1e-2,  # 'Default 1e-2'
	patience = 100, # Default 100, set -1 to train infinitely
	decay_steps = 10000, # Default 10000
	decay_rate = 0.9, # Default 0.9
	pretrained = True,
	# checkpoint = '/datanew/hwb/data/NumReg_Dawei/Number_predict_weights/model-50000.tar', #Name of the pretrained weight file
	# checkpoint='/datanew/hwb/PycharmProjects/SVHN/checkpoint/model-496000.tar',
	checkpoint='/datanew/hwb/data/WG_Num/SVHN/Number_predict_weights/model-696000.tar',
	# Name of the pretrained weight file
	# step=496000,  # Current step, default = False, means to copy the step from the checkpoints
	step = 696000, #Current step, default = False, means to copy the step from the checkpoints
    )

test_cfg = dict(
	mode = 'test', #'directory to test data path',
	batch_size = 128 ,
    cuda = True,
	pretrained = True,
	checkpoint = '/datanew/hwb/data/WG_Num/Negative/Number_predict_weights/model-541000.tar', #Name of the pretrained weight file
	step = 755000	#Current step, default = False, means to copy the step from the checkpoints
    )

eval_cfg = dict(
	mode = 'test', #'directory to test data path',
	batch_size = 128 ,
    cuda = True,
	pretrained = True,
	checkpoint = '/datanew/hwb/data/WG_Num/Negative/Number_predict_weights/model-541000.tar', #Name of the pretrained weight file
	step = 5000	#Current step, default = False, means to copy the step from the checkpoints
    )

loss = dict(overlap_thresh = 0.5,
            prior_for_matching = True,
            bkg_label = 0,
            neg_mining = True,
            neg_pos = 3,
            neg_overlap = 0.5,
            encode_target = False)

optimizer = dict(type='SGD', momentum=0.9, weight_decay=0.0005)

import os
dataset = dict(
    root = '/datanew/hwb/data/WG_Num/SVHN', # The root path of the dataset.
	anno_path = 'Annotations', # path to xml annotations dir
	img_path = 'JPEGImages', # path to img dir
	log_dir = 'Number_predict_weights', # directory to write logs and save checkpoints
	CLASSES = ['__background__','region']
)
