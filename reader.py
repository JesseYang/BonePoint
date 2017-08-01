#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
#         Yukun Chen <cykustc@gmail.com>

import os, sys
import pickle
import numpy as np
from scipy import misc
import random
import six
from six.moves import urllib, range
import copy
import logging
import cv2
from cfgs.config import cfg

from tensorpack import *

def get_data_list(fname):
    with open(fname) as f:
        content = f.readlines()
    rec = [ele.split(' ') for ele in content]
    return rec

def data_enhance(img,bone):
    ori_img_h,ori_img_w = img.shape[0:2]
    x_x = (float(bone[0])+0.5)*ori_img_w
    x_y = (float(bone[1])+0.5)*ori_img_h

    y_x = (float(bone[2])+0.5)*ori_img_w
    y_y = (float(bone[3])+0.5)*ori_img_h


    z_x = (float(bone[4])+0.5)*ori_img_w
    z_y = (float(bone[5])+0.5)*ori_img_h

    w_x = (float(bone[6])+0.5)*ori_img_w
    w_y = (float(bone[7])+0.5)*ori_img_h

    coornidate1_x = int(np.min([x_x,y_x,w_x]))
    coornidate1_y = int(np.min([x_y,y_y,w_y]))

    coornidate2_x = int(np.max([y_x,z_x,w_x]))
    coornidate2_y = int(np.max([y_y,z_y,w_y]))

    width1 = coornidate2_x - coornidate1_x
    height1 = coornidate2_y - coornidate1_y

        #img1 = cv2.rectangle(img, (coornidate1_x,coornidate1_y), (coornidate2_x,coornidate2_y), (255,255,0),1)
    #     cv2.imwrite("1.jpg",img)
      #  plt.imshow(img1)

    new_coornidate1_x = int(random.uniform(0,coornidate1_x))
    new_coornidate1_y = int(random.uniform(0,coornidate1_y))

    new_coornidate2_x = int(random.uniform(coornidate2_x,ori_img_w))
    new_coornidate2_y = int(random.uniform(coornidate2_y,ori_img_h))

    width2 = new_coornidate2_x - new_coornidate1_x
    height2 = new_coornidate2_y - new_coornidate1_y

    x = coornidate1_x - new_coornidate1_x
    y = coornidate1_y - new_coornidate1_y

    co = []
    co.append(x/width2 - 0.5)
    co.append(y/height2 - 0.5)

    co.append((x+width1)/width2 - 0.5)
    co.append(y/height2 -0.5)

    co.append((x+width1)/width2 - 0.5)
    co.append((y+height1)/height2 - 0.5)

    co.append(x/width2 - 0.5)
    co.append((y+height1)/height2 - 0.5)
    
    co = np.asarray(co)
        
    new_imgs = img[new_coornidate1_y:new_coornidate2_y,new_coornidate1_x:new_coornidate2_x]
    
    return [new_imgs,co]


class Data(RNGDataFlow):
    def __init__(self, train_or_test, shuffle=True, affine=False):
        """
        Args:
            train_or_test: string either 'train' or 'test'
            shuffle: default to True
        """
        assert train_or_test in ['train', 'test']
        fname_list = cfg.train_list if train_or_test == "train" else cfg.test_list
        self.train_or_test = train_or_test
        fname_list = [fname_list] if type(fname_list) is not list else fname_list

        self.dp_list = []
        for fname in fname_list:
            self.dp_list.extend(get_data_list(fname))
        self.shuffle = shuffle
        self.affine = affine

    def size(self):
        return len(self.dp_list)
        
    def get_data(self):
        idxs = np.arange(len(self.dp_list))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            dp = self.dp_list[k]
            img_path = dp[0]
            img = cv2.imread(img_path)
            bone = dp[1:]
            
            bone = np.asarray([float(i) for i in bone])
          
            if cfg.anchor_bones != None:
                bone = bone - cfg.anchor_bones

            if cfg.name == "plate" and self.affine == True and random.uniform(0,1) > 0.5:
                [img, bone] = data_enhance(img, bone)
 
            img = cv2.resize(img, (cfg.img_h, cfg.img_w))
            
            yield [img, bone]


if __name__ == '__main__':
    ds = Data('train')
    #import pdb
    #pdb.set_trace()
