from tkinter import *
from tkinter import filedialog
import segment
import generate_model
import predict
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imwrite
import importlib.util

class mainWindow:
    def __init__(self, win):
        intro = 'This is the Deep Learning tool for studying cell morphology'

        self.T = Text(width = 57, height = 10)
        self.T.place(x=20, y=25)
        self.T.insert(END,intro)
        self.settings = {'img_path':'current', 'type':'.tif', 'model':'current', 'input_ch':0, 'segmt_ch':1, 'useGPU':False}
        self.explr_bt = Button(win, text = "Browse Directory", command = self.browseDir)
        self.explr_bt.place(x=100, y=225)
        self.explr_lb = Label(win, text = '/')
        self.explr_lb.place(x=250,y=225)

        self.model_bt = Button(win, text = "Browse Existing Models", command = self.browseFiles)
        self.model_bt.place(x=100, y=250)
        self.model_lb = Label(win, text = '/')
        self.model_lb.place(x=250,y=250)

        self.ftype_lb = Label(win, text='File type')
        self.ftype_lb.place(x=100, y=350)
        self.ftype_en = Entry()
        self.ftype_en.place(x=250, y=350)

        self.input_lbl = Label(win, text='Input Channel')
        self.input_lbl.place(x=100, y=275)
        self.segmt_lbl = Label(win, text='Segment Channel')
        self.segmt_lbl.place(x=100, y=300)

        self.input_ent = Entry(win, width = 2)
        self.input_ent.place(x=250, y=275)
        self.segmt_ent = Entry(win, width = 2)
        self.segmt_ent.place(x=250, y=300)

        self.GPU_lbl = Label(win, text = 'Use GPU')
        self.GPU_lbl.place(x=100, y=325)
        self.ck1 = IntVar()
        self.GPU_chk = Checkbutton(win, variable = self.ck1)
        self.GPU_chk.place(x=250, y=325)

        self.predc_btn = Button(win, text='Predict', command=self.predict)
        self.predc_btn.place(x=100, y=400)
        self.train_btn = Button(win, text='Train a New Model', command=self.open_train)
        self.train_btn.place(x=300, y=400)

    def browseDir(self):
        filename = filedialog.askdirectory(initialdir = "/",
                                          title = "Select a Directory",)
        self.explr_lb.configure(text="Selected Directory: "+filename)
        self.settings['img_path'] = filename
    def browseFiles(self):
        filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a Directory",)
        self.model_lb.configure(text="Selected Model: "+filename)
        self.settings['model'] = filename

    def predict(self):
        self.update_param()
        meta_file = open(self.settings['model'][:-3] + '_metadata.txt', 'r')
        diam = int(meta_file.readline())
        preproc = int(meta_file.readline())
        thres = int(meta_file.readline())
        labels, masks = predict.run_pipeline(self.settings['img_path'], self.settings['type'], self.settings['model'], self.settings['input_ch'], self.settings['segmt_ch'], diam, preproc, self.settings['useGPU'])

        if thres != -1:
            for label in labels:
                label[label > 0.5]=2
                label[label < 0.5]=1
        self.generate_imgs(labels, masks)
        poscount = np.count_nonzero(np.where(label==2, 1, 0))
        negcount = np.count_nonzero(np.where(label==1, 1, 0))
        print('Prediction Complete. Output was saved to supplied directory.')
        print('We identified ' + str(poscount) + ' positive cells and '+ str(negcount) + ' negative cells, for a total of ' + str(poscount+negcount) + ' cells.')



    def generate_imgs(self, labels, masks):
        img_count = 0
        new_masks = []
        images_lst = [file for file in os.listdir(self.settings['img_path'])
                        if file.endswith(self.settings['type'])]
        for mask in masks:
            new_mask = np.zeros(mask.shape)
            label = labels[img_count]
            for cell_id in range(1,int(mask.max())):
                new_mask[mask==cell_id] = label[cell_id-1]
            output_filename = images_lst[img_count][:-4] + '_prediction.tif'
            imwrite(output_filename, new_mask)
            img_count = img_count + 1


    def update_param(self):
        self.settings['type'] = str(self.ftype_en.get())
        self.settings['input_ch'] = int(self.input_ent.get())
        self.settings['segmt_ch'] = int(self.segmt_ent.get())
        self.settings['useGPU'] = bool(self.ck1.get())
    def open_train(self):
        train_win = Toplevel()
        tr_window = training_window(train_win)
        train_win.title('Train a New Model')
        train_win.geometry("700x400+10+10")


class training_window:
    def __init__(self, wind):
        self.settings = {'path':'current', 'type':'.tif', 'input':0, 'truth':1, 'segmt':-1, 'useGPU':False, 'diam':20, 'preproc':0,'modelName':'untitled'}
        self.explr_btn = Button(wind, text = "Browse Directory", command = self.browseFiles)
        self.explr_btn.place(x=100, y=25)
        self.explr_lbl = Label(wind, text = '/')
        self.explr_lbl.place(x=250,y=25)

        self.ftype_lbl = Label(wind, text='File type')
        self.ftype_lbl.place(x=100, y=50)
        self.ftype_ent = Entry(wind)
        self.ftype_ent.place(x=200, y=50)

        self.input_lbl = Label(wind, text='Input Channel')
        self.input_lbl.place(x=100, y=75)
        self.truth_lbl = Label(wind, text='Truth Channel')
        self.truth_lbl.place(x=100, y=100)
        self.segmt_lbl = Label(wind, text='Segment Channel')
        self.segmt_lbl.place(x=100, y=125)

        self.input_ent = Entry(wind, width = 2)
        self.input_ent.place(x=200, y=75)
        self.truth_ent = Entry(wind, width = 2)
        self.truth_ent.place(x=200, y=100)
        self.segmt_ent = Entry(wind, width = 2)
        self.segmt_ent.place(x=200, y=125)

        self.GPU_lbl = Label(wind, text = 'Use GPU')
        self.GPU_lbl.place(x=100, y=150)
        self.ck1 = IntVar()
        self.GPU_chk = Checkbutton(wind, variable = self.ck1)
        self.GPU_chk.place(x=200, y=150)

        self.diam_lbl = Label(wind, text = 'Estimated Cell Diameter')
        self.diam_lbl.place(x=50, y=175)
        self.diam_ent = Entry(wind, width = 3)
        self.diam_ent.place(x=200, y=175)

        self.prepr_lbl = Label(wind, text = 'Preprocessor type')
        self.prepr_lbl.place(x=20, y=200)
        self.prepr_ent = Entry(wind, width = 3)
        self.prepr_ent.place(x=200, y=200)

        self.model_lbl = Label(wind, text = 'Name for trained model?')
        self.model_lbl.place(x=20, y=250)
        self.model_ent = Entry(wind)
        self.model_ent.place(x=200, y=250)

        self.train_btn = Button(wind, text='Train Model', command=self.open_segmt)
        self.train_btn.place(x=200, y=300)
        self.exit_btn = Button(wind, text='Exit', command=wind.destroy)
        self.exit_btn.place(x=500, y=300)


    def browseFiles(self):
        filename = filedialog.askdirectory(initialdir = "/",
                                          title = "Select a Directory",)
        self.explr_lbl.configure(text="Selected Directory: "+filename)
        self.settings['path'] = filename


    def update_param(self):
        self.settings['type'] = str(self.ftype_ent.get())
        self.settings['input'] = int(self.input_ent.get())
        self.settings['truth'] = int(self.truth_ent.get())
        self.settings['segmt'] = int(self.segmt_ent.get())
        self.settings['useGPU'] = bool(self.ck1.get())
        self.settings['diam'] = int(self.diam_ent.get())
        self.settings['preproc'] = int(self.prepr_ent.get())
        self.settings['modelName'] = str(self.model_ent.get())
    def open_segmt(self):
        self.update_param()
        data_x, data_y = segment.run_pipeline(self.settings['path'], self.settings['type'], self.settings['input'], self.settings['truth'], self.settings['segmt'], self.settings['useGPU'], self.settings['diam'], self.settings['preproc'])
        cellCount = data_y.shape[0]
        tl = Toplevel()
        pg_window = progresswin(tl, cellCount, data_x, data_y, self.settings)
        tl.title('Training model')
        tl.geometry('300x300')
        self.generate_histogram(data_y)

    def generate_histogram(self, data):
        plt.hist(data, bins = 100)
        plt.xlabel('Mean Fluorescence Intensity')
        plt.ylabel('Cell Count')
        plt.show()

class progresswin:
    def __init__(self, Topl, cellCount, data_x, data_y, settings):
        self.data_x = data_x
        self.data_y = data_y
        self.settings = settings
        self.text = StringVar()
        self.text.set('')
        self.thres = 0

        self.label1 = Label(Topl, text = 'We identified '+str(cellCount) + ' cells.')
        self.label1.place(x=20, y=100)
        self.label2 = Label(Topl, text = 'Choose threshold to see positive count')
        self.label2.place(x=20, y=125)

        self.thres_ent = Entry(Topl, width = 5)
        self.thres_ent.place(x=250, y=150)

        self.label3 = Label(Topl, text = 'Threshold for binary classification')
        self.label3.place(x=20, y=150)

        self.count_btn = Button(Topl, text='Count', command = self.count)
        self.count_btn.place(x=20, y=175)

        self.cont_btn = Button(Topl, text='Continue', command=self.run)
        self.cont_btn.place(x=50, y=200)

        self.exit_btn = Button(Topl, text='Exit',command=Topl.destroy)
        self.exit_btn.place(x=200, y=200)

    def count(self):
        self.update_param()
        if self.thres != -1:
            data_bin = self.binarize_label(self.data_y, self.thres)
        poscount = np.count_nonzero(data_bin)
        self.label2.config(text = 'Of these, ' +str(poscount) + ' cells were positive.')

    def run(self):
        self.update_param()
        if self.thres != -1:
            self.data_y = self.binarize_label(self.data_y, self.thres)
        x = generate_model.init_model(self.settings['diam']*2, self.thres)
        generate_model.fit_model(x, self.data_x, self.data_y, self.settings['diam']*2, self.settings['modelName'])
        file = open(self.settings['modelName'] + '_metadata.txt', 'w')
        file.write(str(self.settings['diam']) + '\n')
        file.write(str(self.settings['preproc']) + '\n')
        file.write(str(self.thres) + '\n')
        file.write(str(self.settings['input']) + '\n')
        file.write(str(self.settings['truth']) + '\n')
        file.write(str(self.settings['segmt']) + '\n')
        file.write(str(self.settings['useGPU']) + '\n')
        file.write(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + '\n')

    def update_param(self):
        self.thres = int(self.thres_ent.get())
    def binarize_label(self, y, t):
        x = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            if y[i] >= t:
                x[i] = 1
            else:
                x[i] = 0
        return x


def check_package_installation():
    packages_required = ['numpy',
                         'matplotlib.pyplot',
                         'cellpose',
                         'skimage',
                         'sklearn',
                         'tensorflow',
                         'keras',
                         'torch'
    ]
    missing = False
    for package in packages_required:
        if importlib.util.find_spec(package)==None:
            print(package + ' is not installed')
            missing = True
    if missing:
        print('Required packages not installed. Exiting...')
        sys.exit()
    else:
        print('All packages are present. Opening the GUI...')

check_package_installation()

window=Tk()
mywin=mainWindow(window)
window.title('Deep Learning Cell Classifier Trainer')
window.geometry("500x500+10+10")
window.mainloop()
