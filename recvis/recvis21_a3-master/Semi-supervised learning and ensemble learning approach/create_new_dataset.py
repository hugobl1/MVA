#!/usr/bin/python

import sys, os, glob, shutil


out_dir = 'bird_dataset_v2/train_images/'

to_dirs = os.listdir(out_dir)

n=len(npnames)
for i in range(n):
    shutil.move(npnames[i], out_dir+to_dirs[npclasses[i]])