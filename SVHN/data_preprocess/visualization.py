import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def draw_scores_of_checkpoints(scores ,mode , label_type, file_dir=None,file=None,read_file=None):

    title = '{}_{}'.format(mode, label_type)

    '''directory to save the figure and json file ! '''
    dir = os.path.join(file_dir,mode, 'SVHN_scores')

    if not os.path.exists(dir):
        os.makedirs(dir)

    if read_file :
        with open(read_file,'r') as f :
            draw = json.load(f)
    else:
        draw = {}
        if len(scores[0]) > 0:
            draw['correct_length'] = scores[0][:,0].tolist()
            draw['correct_digit 0'] = scores[0][:,1].tolist()
            draw['correct_digit 1'] = scores[0][:,2].tolist()
            draw['correct_sum_score'] = (scores[0][:,0] + scores[0][:,1] + scores[0][:,2]).tolist()

        if len(scores[1]) > 0:
            draw['wrong_length'] = scores[1][:,0].tolist()
            draw['wrong _digit 0'] = scores[1][:,1].tolist()
            draw['wrong_digit 1'] = scores[1][:,2].tolist()
            draw['wrong_sum_score'] = (scores[1][:,0] + scores[1][:,1] + scores[1][:,2]).tolist()

        with open(os.path.join(dir, title + '.json'),'w') as fw:
            json.dump(draw,fw)

    '''set label range for each one! '''
    range_list = [(0,25),(0,31),(0,20),(0,70),(0,25),(0,31),(0,20),(0,70)]

    plt.figure(figsize=(100,20)) # set figure size
    plt.rcParams.update({'font.size': 60}) # set font size
    for index,name in enumerate(draw):
        plt.subplot(2, 4, index+1)
        his = np.array(draw[name])
        scale = np.histogram(his,bins=500,range = range_list[index])

        num = 'num:{:->8}\n'.format(len(his))
        max_score = 'max:{:.4f},'.format(np.max(his))
        min_score = 'min:{:.4f}\n'.format(np.min(his))
        mean_score = '(r)mean:{:.4f},'.format(np.mean(his))
        median_score = '(g)median:{:.4f}'.format(np.median(his))

        plt.hist(his,bins=500,range=range_list[index])
        '''draw mean and median line in the scores histogram'''
        plt.axvline(x=np.mean(his),ymin=np.min(scale[0]),ymax=np.max(scale[0]),linewidth=5,color='r')
        plt.axvline(x=np.median(his),ymin=np.min(scale[0]),ymax=np.max(scale[0]),linewidth=5,color='g')

        plt.title(name)
        plt.ylabel('count')
        plt.xlabel(num + max_score+min_score+mean_score + median_score)
        plt.grid(True)

    plt.subplots_adjust(hspace=0.5) # set gap between subplot !
    plt.tight_layout()
    plt.savefig(os.path.join(dir, title + '.png'))
    plt.close()
    print('Figure in ',dir, ' saved !')
