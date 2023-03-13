
from cellpose import models
from cellpose import plot
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
from skimage import io
import segment
from tensorflow import keras
import torch

def generate_input_for_pred(path_str, file_type, input_ch, segmt_ch):
    images_lst = [file for file in os.listdir(path_str)
                    if file.endswith(file_type)]
    bf_imgs = []
    sg_imgs = []
    for filenames in images_lst:
        im = io.imread(os.path.join(path_str,filenames))
        im = dimension_compat(im)
        bf_imgs.append(im[:,:,input_ch])
        if segmt_ch != -1:
            sg_imgs.append(im[:,:,segmt_ch])
    return bf_imgs, sg_imgs

def normalize_bf(img):
    avg = np.mean(img)
    img = img - avg
    std = np.std(img)
    return img/std

def dimension_compat(im):
    shape = im.shape
    if len(shape)==2:
        im = np.expand_dims(im, axis = 2)
    elif shape[0] < shape[2]:
        print('Your image looks like channel is listed first. Transposing...')
        im = np.transpose(im, axes = (1,2,0))
    return im

def generate_segments(bf_img, sg_img = 0, use_GPU = False, diameter = 20):
    model = models.Cellpose(gpu=use_GPU, model_type = 'cyto2', device=torch.device('cuda'))
    segment_bf = []

    if sg_img != []:
        segment_img = sg_img
    else:
        segment_img = bf_img
    mask, flows, styles, diams = model.eval(segment_img, diameter=diameter, flow_threshold=0.2, channels=[0,0])

    maxnum_cell = mask.max()
    bf_img = add_padding(normalize_bf(bf_img), int(diameter*1.5))
    mask = add_padding(mask, int(diameter*1.5))
    for j in range(1, maxnum_cell):
        x,y = np.where(mask==j)
        min_indx = int(x.min()-diameter/2)
        max_indx = int(x.min()+diameter*1.5)
        min_indy = int(y.min()-diameter/2)
        max_indy = int(y.min()+diameter*1.5)
        segment_bf.append(bf_img[min_indx:max_indx, min_indy:max_indy])
    return segment_bf, mask
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

def run_model(segment_bf, model):
    chosen_model = keras.models.load_model(model)
    output = chosen_model.predict(segment_bf)
    return output

def run_pipeline(path_str, file_type, model_name, input_ch, segmt_ch, diam, preproc, useGPU):
    bf_imgs, sg_imgs = generate_input_for_pred(path_str, file_type, input_ch, segmt_ch)
    masks = []
    labels = []
    for i in range(len(bf_imgs)):
        if segmt_ch==-1:
            segment_bf, mask = generate_segments(bf_imgs[i], [], use_GPU = useGPU, diameter = diam)
        else:
            segment_bf, mask = generate_segments(bf_imgs[i], sg_imgs[i], use_GPU = useGPU, diameter = diam)
        masks.append(mask)
        segment_bf = np.stack(segment_bf,axis=0)
        segment_bf = segment_bf.reshape(-1, diam*2, diam*2, 1)
        labels.append(run_model(segment_bf, model_name))

    return labels, masks
