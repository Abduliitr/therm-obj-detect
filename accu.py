#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 11:02:00 2020

@author: atmospheric
"""


import sys
import time
import argparse

import cv2
import pycuda.autoinit

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization



WINDOW_NAME = 'TrtSsdDemo'
INPUT_HW = (300, 300)

model="ssd_mobilenet_v1_coco"

import glob
mylist=(glob.glob("/home/atmospheric/Downloads/Thermal Dataset/webcam/*"))
path="/home/atmospheric/Downloads/Thermal Dataset/webcam/"
print(mylist[0])
file1 = open("myfile.txt","w") 
t=time.time()
for i,image in enumerate(mylist):
	
	img = cv2.imread(image)
	cls_dict = get_cls_dict(model.split('_')[-1])
	trt_ssd = TrtSSD(model, INPUT_HW)
	conf_th=0.3
	boxes, confs, clss = trt_ssd.detect(img, conf_th)
	print((i/len(mylist))*100)
	
print(time.time()-t)
