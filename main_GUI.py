from tkinter import *
from tkinter import filedialog
import segment
import generate_model
import argparse
import os

class mainWindow:
    def __init__(self, win):
        intro = 'Hi. This is the Deep Learning tool for studying cell morphology'

        self.T = Text()
        self.T.place(x=20, y=25)
        self.T.insert(END,intro)
        self.settings = {'img_path':'current', 'type':'.tif', 'model':'current'}
        self.explr_bt = Button(win, text = "Browse Directory", command = self.browseDir)
        self.explr_bt.place(x=100, y=225)
        self.explr_lb = Label(win, text = '/')
        self.explr_lb.place(x=250,y=225)

        self.model_bt = Button(win, text = "Browse Existing Models", command = self.browseFiles)
        self.model_bt.place(x=100, y=250)
        self.model_lb = Label(win, text = '/')
        self.model_lb.place(x=250,y=250)

        self.ftype_lb = Label(win, text='File type')
        self.ftype_lb.place(x=100, y=300)
        self.ftype_en = Entry()
        self.ftype_en.place(x=200, y=300)

        self.predc_btn = Button(win, text='Predict', command=self.predict)
        self.predc_btn.place(x=100, y=350)
        self.train_btn = Button(win, text='Train a New Model', command=self.open_train)
        self.train_btn.place(x=300, y=350)
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
        print('placeholder')

    def open_train(self):
        train_win = Toplevel()
        tr_window = training_window(train_win)
        train_win.title('Train a New Model')
        train_win.geometry("700x400+10+10")

class training_window:
    def __init__(self, wind):
        self.settings = {'path':'current', 'type':'.tif', 'input':0, 'truth':1, 'segmt':-1, 'useGPU':False, 'diam':20, 'threshold':-1,'modelName':'untitled'}
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

        self.thrsh_lbl = Label(wind, text = 'Threshold for Positive Label')
        self.thrsh_lbl.place(x=20, y=200)
        self.thrsh_ent = Entry(wind, width = 8)
        self.thrsh_ent.place(x=200, y=200)

        self.model_lbl = Label(wind, text = 'Name for trained model?')
        self.model_lbl.place(x=20, y=250)
        self.model_ent = Entry(wind)
        self.model_ent.place(x=200, y=250)

        self.train_btn = Button(wind, text='Train Model', command=self.open_segmt)
        self.train_btn.place(x=200, y=300)
        self.exit_btn = Button(wind, text='Exit', command=wind.destroy)
        self.exit_btn.place(x=500, y=300)
        # self.b2=Button(win, text='Subtract')
        # self.b2.bind('<Button-1>', self.sub)
        # self.b1.place(x=100, y=150)
        # self.b2.place(x=200, y=150)
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
        self.settings['threshold'] = int(self.thrsh_ent.get())
        self.settings['modelName'] = str(self.model_ent.get())
    def open_segmt(self):
        self.update_param()
        data_x, data_y = segment.run_pipeline(self.settings['path'], self.settings['type'], self.settings['input'], self.settings['truth'], self.settings['segmt'], self.settings['useGPU'], self.settings['diam'], self.settings['threshold'])
        cellCount = data_y.shape[0]
        tl = Toplevel()
        pg_window = progresswin(tl, cellCount, data_x, data_y, self.settings['diam'], self.settings['threshold'], self.settings['modelName'])
        tl.title('Training model')
        tl.geometry('300x300')


class progresswin:
    def __init__(self, Topl, cellCount, data_x, data_y, diam, thres, modN):
        self.data_x = data_x
        self.data_y = data_y
        self.diam = diam
        self.thres = thres
        self.modN = modN
        self.label1 = Label(Topl, text = 'We identified '+str(cellCount) + ' cells. Do you wish to continue?')
        self.label1.place(x=20, y=100)

        self.cont_btn = Button(Topl, text='Continue', command=self.run)
        self.cont_btn.place(x=50, y=200)

        self.exit_btn = Button(Topl, text='Exit',command=Topl.destroy)
        self.exit_btn.place(x=200, y=200)

    def run(self):
        x = generate_model.init_model(self.diam*2, self.thres)
        generate_model.fit_model(x, self.data_x, self.data_y, self.diam*2, self.modN)



window=Tk()
mywin=mainWindow(window)
window.title('Deep Learning Cell Classifier Trainer')
window.geometry("500x400+10+10")
window.mainloop()
