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

class Data(RNGDataFlow):
    def __init__(self, train_or_test, shuffle=True):
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
            img = cv2.resize(img, (cfg.img_h, cfg.img_w))
            bone = dp[1:]
            bone = np.asarray([float(ele) for ele in bone])
            if cfg.anchor_bones != None:
                bone = bone - cfg.anchor_bones
            yield [img, bone]


if __name__ == '__main__':
    ds = Data('train')
    import pdb
    pdb.set_trace()
