import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_files', help='comma separated data file list', required=True)
args = parser.parse_args()

data_files = args.data_files.split(',')

bones = []
for data_file in data_files:
	with open(data_file) as f:
		content = f.readlines()
	records = [ele.split(' ')[1:] for ele in content]
	records = [np.asarray([float(e) for e in ele]) for ele in records]
	bones.extend(records)

bones = np.asarray(bones)
anchor_bone = np.mean(bones, axis=0)
print(anchor_bone)