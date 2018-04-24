import os
from pathlib import Path

import numpy
from PIL import Image
import six

import numpy as np

from io import BytesIO
import os
import pickle
import json
import numpy as np

import skimage.io as io

from chainer.dataset import dataset_mixin

# download `BASE` dataset from http://cmp.felk.cvut.cz/~tylecr1/facade/
class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir='./facade/base', data_range=(1,300)):
        print("load dataset start")
        print("    from: %s"%dataDir)
        print("    range: [%d, %d)"%(data_range[0], data_range[1]))
        dataDir = Path(dataDir)
        self.dataDir = dataDir
        self.dataset = []
        for i in range(data_range[0],data_range[1]):
            img = Image.open(str(dataDir / ("cmp_b%04d.jpg"%i)))
            label = Image.open(str(dataDir / ("cmp_b%04d.png"%i)))
            w,h = img.size
            r = 286 / float(min(w,h))
            # resize images so that min(w, h) == 286
            img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
            label = label.resize((int(r*w), int(r*h)), Image.NEAREST)
            
            img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
            label_ = np.asarray(label)-1  # [0, 12)
            label = np.zeros((12, img.shape[1], img.shape[2])).astype("i")
            for j in range(12):
                label[j,:] = label_==j
            self.dataset.append((img,label))
        print("load dataset done")
    
    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i, crop_width=256):
        _,h,w = self.dataset[i][0].shape
        x_l = np.random.randint(0,w-crop_width)
        x_r = x_l+crop_width
        y_l = np.random.randint(0,h-crop_width)
        y_r = y_l+crop_width
        return self.dataset[i][1][:,y_l:y_r,x_l:x_r], self.dataset[i][0][:,y_l:y_r,x_l:x_r]

class MugFaceDataset(dataset_mixin.DatasetMixin):
    def __init__(self, path):
        self.dataset = []
        self.depth_frames = list(path.glob("**/user*/depth/*.jpg"))
        self.rgb_frames = list(path.glob("**/user*/rgb/*.jpg"))
    
    def __len__(self):
        return len(self.rgb_frames)

    # return (label, img)
    def get_example(self, i, crop_width=256):
        img = Image.open(str(self.depth_frames[i]))
        label = Image.open(str(self.rgb_frames[i]))

        img = np.asarray(img).astype("f")
        img = np.expand_dims(img, axis=2)
        img = img.transpose(2,0,1)/128.0-1.0
        label = np.asarray(label).astype("f").transpose(2,0,1)/128.0-1.0

        return img, label
    
