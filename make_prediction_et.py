
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 9 18:18:03 2017

@author: insane
"""

'''
Title           :make_predictions_et.py
'''

import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import lmdb
caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
'''
A simple function to apply Histogram Equalization and then scale down the image for faster computation
'''
    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def transimg(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
'''
Similar to the above function but only aims at images with single channel
'''
    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
  
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

def mak_datum(data, label):
'''
Something to hold the data
'''    
        return caffe.proto.caffe_pb2.Datum(
           channels = 1,
           height = 1,
           width = 205,
           data = data.tostring(),
           label = int(label))

'''
Reading mean image, caffe model and its weights 
'''
#Read mean image

'''
A pretrained netowork called 'placesCNN205' was made use of for this project. The network was trained by MIT on 2.5 million images and can classify images into 205 classes. 
The pretrained network is used for classifying another dataset which has only 8 classes and a total of 2688 images. 1888 images is used as training set and the remaining
800 images makes the test set. 
'''
#Read model architecture and trained model's weights
net = caffe.Net('/home/insane/caffe/models/placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt',
                '/home/insane/caffe/models/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel',
                caffe.TEST)

#Define image transformers
transform = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
#Reading image paths
test_img_paths = [img_path for img_path in glob.glob("/home/insane/Documents/data/test2/*.jpg")]
dest_path ='/home/insane/Documents/data/txt/et/test.txt'
dest_path_label ='/home/insane/Documents/data/txt/et/test_label.txt'

#Making predictions
test_ids = []
preds = []
predcom = []
data_mat = np.zeros((1888,205),dtype='float32')             #1888 refers to the number of images in a training set. change to 800 for testing set.  
label_mat = np.zeros((1888,1), dtype='int')                 # creating two variables to save 205 features and the labels respectively.

for ind, each_img_path in enumerate(test_img_paths):
'''
Assigning labels to the images using their image names as condition. 
'''
  if 'opencountry' in each_img_path:
    label = 0 
    img = cv2.imread(each_img_path, cv2.IMREAD_COLOR)
    img = transimg(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    image = 0.3*r + 0.59*g + 0.11*b                        #This line here converts RGB image to greyscale. With use of some codes, RGB can be coverted to other planes
    net.blobs['data'].data[...] = transform.preprocess('data', image)
    out = net.forward()
    pred_probas = out['prob']                              #A normal classifier will output the class with the maximum probability. Here instead of maximum, all probability values are taken into consideration to form a vector.
  elif 'coast' in each_img_path:
    label = 1
    img = cv2.imread(each_img_path, cv2.IMREAD_COLOR)
    img = transimg(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    image = 0.3*r + 0.59*g + 0.11*b
    net.blobs['data'].data[...] = transform.preprocess('data', image)
    out = net.forward()
    pred_probas = out['prob']
  elif'forest' in each_img_path:
    label = 2
    img = cv2.imread(each_img_path, cv2.IMREAD_COLOR)
    img = transimg(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    image = 0.3*r + 0.59*g + 0.11*b
    net.blobs['data'].data[...] = transform.preprocess('data', image)
    out = net.forward()
    pred_probas = out['prob']
  elif 'highway' in each_img_path:
    label = 3
    img = cv2.imread(each_img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']
  elif 'insidecity' in each_img_path:
    label = 4
    img = cv2.imread(each_img_path, cv2.IMREAD_COLOR)
    img = transimg(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    image = 0.3*r + 0.59*g + 0.11*b
    net.blobs['data'].data[...] = transform.preprocess('data', image)
    out = net.forward()
    pred_probas = out['prob']
  elif 'street' in each_img_path:
    label = 5                                                  #My project involves utilization of many color spaces. Images in this particular class are RGB 
    img = cv2.imread(each_img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']
  elif 'mountain' in each_img_path:
    label = 6
    img = cv2.imread(each_img_path, cv2.IMREAD_COLOR)          # This class has images which are converted from RGB to YCbCr.
    image = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    r=image[:,:,0]
    g=image[:,:,1]
    b=image[:,:,2]
    Ey = 0.299*r + 0.587*g + 0.114*b
    Ecb = 0.5*b -0.169*r - 0.331*g
    Ecr = 0.5*r - 0.419*g - 0.081*b 
    y = Ey + 16
    cb = Ecb + 128
    cr = Ecr + 128
    image2 = np.dstack((y,cb,cr))
    net.blobs['data'].data[...] = transformer.preprocess('data', image2)
    out = net.forward()
    pred_probas = out['prob']
  elif 'tallbuildings'in each_img_path:
    label = 7
    img = cv2.imread(each_img_path, cv2.IMREAD_COLOR)
    img = transimg(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    image = 0.3*r + 0.59*g + 0.11*b
    net.blobs['data'].data[...] = transform.preprocess('data', image)
    out = net.forward()
    pred_probas = out['prob']                                 # By the time the code reaches the last image, this variable will be a tuple of 1888 times 205 in dimension 
  data_mat[ind,:]= pred_probas
  predcom.append(pred_probas)
  label_mat[ind,:] = label    
  train_ids = train_ids + [img_path.split('/')[-1][:-4]]
  preds = preds + [pred_probas]


np.savetxt(dest_path,data_mat)
np.savetxt(dest_path_label,label_mat)                        #Saving the feature matrix and the corresponding labels obtained as text files for training a extra tree classifier. 
