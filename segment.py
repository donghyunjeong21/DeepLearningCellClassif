# Documentation:
# Authors: Donghyun Paul Jeong, Maksym Zarodiuk
#
# This file contains the python functions necessary to generate cell segments based on
# input images.


from cellpose import models
from cellpose import plot
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
from skimage import io
import torch

import sys
np.set_printoptions(threshold=sys.maxsize)

# Function to generate a list of images to be pulled from user-selected directory.
# Takes in the following: path to input file in str type, file type in str, and channel numbers
# input_ch = brightfield channel to be used as input
# truth_ch = fluorescent channel to be used to generate labels
# segmt_ch (optional, -1 indicates no segment channel) = separate fluorescent channel for segmentation purpose

def generate_imgs(path_str, file_type = '.tif', input_ch = 0, truth_ch = 1, segmt_ch = -1):
    images_lst = [file for file in os.listdir(path_str)
                    if file.endswith(file_type)]
    bf_imgs = []
    fl_imgs = []
    sg_imgs = []
    for filenames in images_lst:
        im = io.imread(os.path.join(path_str,filenames))
        im = dimension_compat(im)
        bf_imgs.append(im[:,:,input_ch])
        fl_imgs.append(im[:,:,truth_ch])
        if segmt_ch != -1:
            sg_imgs.append(im[:,:,segmt_ch])
    return bf_imgs, fl_imgs, sg_imgs

# Checks if dimensions are comptatible, and if not, transposes
def dimension_compat(im):
    shape = im.shape
    if shape[0] < shape[2]:
        print('Your image looks like channel is listed first. Transposing...')
        im = np.transpose(im, axes = (1,2,0))
    return im

def preprocessing(imgs, type_of_preprocess):
    #Placeholder
    return imgs

# Generates segments based on Cellpose segmentation results
# Takes either the brightfield channel or the segmentation channel and segments.
# Then creates a list of square images of size diameter*2 containing pixel values for each cell
# Pixels in each segment that is not part of that cell is set to 0
def generate_segments(bf_imgs, fl_imgs, sg_imgs = [], use_GPU = False, diameter = 20):
    masks = []
    model = models.Cellpose(gpu=use_GPU, model_type = 'cyto2', device=torch.device('cuda'))
    segment_bf = []
    segment_fl = []

    for i in range(len(bf_imgs)):
        bf_img = bf_imgs[i]
        fl_img = fl_imgs[i]
        if len(sg_imgs) != 0:
            segment_img = sg_imgs[i]
        else:
            segment_img = bf_img
        mask, flows, styles, diams = model.eval(segment_img, diameter = None, flow_threshold=0.2, channels=[0,0])
        # masks.append(mask)
        maxnum_cell = mask.max()
        bf_img = add_padding(bf_img, int(diameter*1.5))
        fl_img = add_padding(fl_img, int(diameter*1.5))
        mask = add_padding(mask, int(diameter*1.5))

        for j in range(1, maxnum_cell):
            x,y = np.where(mask==j)
            bin_mask = np.where(mask == j, 1, 0)

            min_indx = int(x.min()-diameter/2)
            max_indx = int(x.min()+diameter*1.5)
            min_indy = int(y.min()-diameter/2)
            max_indy = int(y.min()+diameter*1.5)
            segment_bf.append(bf_img[min_indx:max_indx, min_indy:max_indy])
            segment_fl.append(np.multiply(fl_img[min_indx:max_indx, min_indy:max_indy], bin_mask[min_indx:max_indx, min_indy:max_indy]))
    return segment_bf, segment_fl, len(segment_bf)

# A function to add padding to edges to make sure all output is of same dimension
def add_padding(im, d):
    xdim, ydim = im.shape
    leftpad = np.zeros([xdim, d])
    y = np.append(im, leftpad, axis = 1)
    y = np.append(leftpad, y, axis = 1)
    xdim, ydim = y.shape
    bottompad = np.zeros([d, ydim])
    z = np.append(y, bottompad, axis = 0)
    z = np.append(bottompad, z, axis = 0)
    return z

# Generates labels. If threshold = -1, then it gives mean intensity in truth channel.
# If threshold is non- -1, then it labels segment as 0 if under threshold, 1 if over
def generate_labels(segment_fl):
    label = np.zeros(len(segment_fl))
    for i in range(len(segment_fl)):
        label[i] = np.sum(segment_fl[i])/np.count_nonzero(segment_fl[i])
    return label

def run_pipeline(img_dir, file_type, input_ch, truth_ch, segmt_ch, use_GPU, diameter, preproc):
    if img_dir == 'current':
        img_dir = pathlib.Path(__file__).parent.resolve()
    bf_imgs, fl_imgs, sg_imgs = generate_imgs(img_dir, file_type = file_type, input_ch = input_ch, truth_ch = truth_ch, segmt_ch = segmt_ch)
    bf_imgs = preprocessing(bf_imgs, preproc)
    print('Imported the images. Segmenting them...')
    segment_bf, segment_fl, numcell = generate_segments(bf_imgs, fl_imgs, use_GPU = use_GPU, diameter = diameter)
    print('Segmented the images. Identified '+ str(numcell) + ' cells across all the supplied images. Generating labels...')
    label = generate_labels(segment_fl)
    print('Generated labels. Training model now...')
    return segment_bf, label
