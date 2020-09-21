import os
import os.path as osp

import random
import xml.etree.ElementTree as ET 
import cv2 
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

import itertools
from math import sqrt
import time 

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

import warnings
warnings.filterwarnings("ignore")

cfg = {
    "num_classes": 21,
    "input_size": 300,
    "bbox_aspect_num": [4, 6, 6,6,4,4], #source1->6
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300], #df box size
    "min_size": [30,150,11,162,213,315],
    "max_size": [60,111,162,213,264,315],
    "aspect_ratios": [[2],[2,3],[2,3],[2,3],[2],[2]]
}