import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.name = 'sealion'
cfg.obj_num = 4
cfg.bone_num = 13
cfg.match = [1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 3, 3]

cfg.img_w = 224
cfg.img_h = 224

cfg.anchor_bones = [0.42506769, 0.25954831, 0.1337511, 0.31739664, -0.09117149, 0.28229989,
					-0.20508224, 0.13632212, -0.19473236, -0.05591155, -0.14091084, -0.22759439,
					-0.18862732, -0.45212805, -0.00121401, 0.00345235, 0.14096452, 0.37749384,
					-0.79428331, 0.12560712, 0.10606102, 0.06892555, -0.47056319, -0.27855245,
					0.02235486, 0.23543663]

cfg.train_list = ["sealion_train.txt"]
cfg.test_list = "sealion_test.txt"
