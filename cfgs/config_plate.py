import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.name = 'plate'
cfg.obj_num = 1
cfg.bone_num = 4
cfg.match = [0, 0, 0, 0]

cfg.img_w = 224
cfg.img_h = 224

cfg.anchor_bones = None

cfg.weight_decay = 1e-4

cfg.train_list = [cfg.name + "_train.txt"]
cfg.test_list = cfg.name + "_test.txt"
