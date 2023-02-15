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
        print('Your video looks like channel is listed first. Transposing...')
        im = np.transpose(im, axes = (1,2,0))
    return im

# Generates segments based on Cellpose segmentation results
# Takes either the brightfield channel or the segmentation channel and segments.
# Then creates a list of square images of size diameter*2 containing pixel values for each cell
# Pixels in each segment that is not part of that cell is set to 0

def generate_segments(bf_imgs, fl_imgs, sg_imgs = [], use_GPU = False, diameter = 20):
    masks = []
    model = models.Cellpose(gpu=use_GPU, model_type = 'cyto')
    segment_bf = []
    segment_fl = []

    for i in range(len(bf_imgs)):
        bf_img = bf_imgs[i]
        fl_img = fl_imgs[i]
        if len(sg_imgs) != 0:
            segment_img = sg_imgs[i]
        else:
            segment_img = bf_img
        mask, flows, styles, diams = model.eval(segment_img, diameter=diameter, flow_threshold=0.6, channels=[0,0])
        masks.append(mask)
        maxnum_cell = mask.max()
        sgmt_dim = 2
        bf_img = add_padding(bf_img, sgmt_dim*diameter)
        fl_img = add_padding(fl_img, sgmt_dim*diameter)

        for j in range(1, maxnum_cell):
            x,y = np.where(mask==j)
            bin_mask = add_padding(np.where(mask ==j, 1, 0), sgmt_dim*diameter)
            xmin = x.min()
            ymin = y.min()

            segment_bf.append(np.multiply(bf_img[xmin:xmin+diameter*2,ymin:ymin+diameter*2], bin_mask[xmin:xmin+diameter*2,ymin:ymin+diameter*2]))
            segment_fl.append(np.multiply(fl_img[xmin:xmin+diameter*2,ymin:ymin+diameter*2], bin_mask[xmin:xmin+diameter*2,ymin:ymin+diameter*2]))
    return segment_bf, segment_fl

# A function to add padding to edges to make sure all output is of same dimension
def add_padding(im, d):
    xdim, ydim = im.shape
    leftpad = np.zeros([xdim, d])
    y = np.append(im, leftpad, axis = 1)
    xdim, ydim = y.shape
    bottompad = np.zeros([d, ydim])
    z = np.append(y, bottompad, axis = 0)
    return z

# Generates labels. If threshold = -1, then it gives mean intensity in truth channel.
# If threshold is non- -1, then it labels segment as 0 if under threshold, 1 if over

def generate_labels(segment_fl, threshold = -1):
    label = np.zeros(len(segment_fl))
    for i in range(len(segment_fl)):
        label[i] = np.sum(segment_fl[i])/np.count_nonzero(segment_fl[i])
    if threshold != -1:
        label[label < threshold] = 0
        label[label >= threshold] = 1
    return label

# A code to test. Should be deleted in the final version
def testrun():
    img_dir = pathlib.Path(__file__).parent.resolve().__str__()
    print(type(img_dir))
    bf_imgs, fl_imgs, sg_imgs = generate_imgs(img_dir, input_ch = 1, truth_ch = 0)
    segment_bf, segment_fl = generate_segments(bf_imgs, fl_imgs, use_GPU = True, diameter = 20)
    label = generate_labels(segment_fl, threshold = 2000)
    return segment_bf, label
