from cellpose import models
from cellpose import plot
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
from skimage import io

#Setting the path where images are stored
img_dir = pathlib.Path(__file__).parent.resolve().__str__()
print(type(img_dir))

def generate_imgs(path_str, file_type = '.tif', input_ch = 0, truth_ch = 1, segmt_ch = -1):
    images_lst = [file for file in os.listdir(path_str)
                    if file.endswith(file_type)]
    print(images_lst)
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

def dimension_compat(im):
    shape = im.shape
    if shape[0] < shape[2]:
        print('Your video looks like channel is listed first. Transposing...')
        im = np.transpose(im, axes = (1,2,0))
    return im

def generate_segments(bf_imgs, fl_imgs, sg_imgs = [], use_GPU = False, diameter = 20):

    masks = []
    print(use_GPU)
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
        for j in range(1, maxnum_cell):
            x,y = np.where(mask==j)
            bin_mask = np.where(mask ==j, 1, 0)
            xmin = x.min()
            ymin = y.min()


            segment_bf.append(np.multiply(bf_img[xmin:xmin+diameter*2,ymin:ymin+diameter*2], bin_mask[xmin:xmin+diameter*2,ymin:ymin+diameter*2]))
            segment_fl.append(np.multiply(fl_img[xmin:xmin+diameter*2,ymin:ymin+diameter*2], bin_mask[xmin:xmin+diameter*2,ymin:ymin+diameter*2]))
    plt.imshow(segment_fl[25], cmap='gray')
    plt.show()

    return segment_bf, segment_fl

def generate_labels(segment_fl, threshold = -1):
    label = np.zeros(len(segment_fl))
    for i in range(len(segment_fl)):
        label[i] = np.sum(segment_fl[i])/np.count_nonzero(segment_fl[i])
    if threshold != -1:
        label[label < threshold] = 0
        label[label >= threshold] = 1
    return label

bf_imgs, fl_imgs, sg_imgs = generate_imgs(img_dir, input_ch = 1, truth_ch = 0)
segment_bf, segment_fl = generate_segments(bf_imgs, fl_imgs, use_GPU = True, diameter = 20)
label = generate_labels(segment_fl, threshold = 2000)
print(label)
