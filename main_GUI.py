from tkinter import *
from tkinter import filedialog
import segment
import generate_model
import argparse
import os

class MyWindow:
    def __init__(self, win):
        self.settings = {'path':'current', 'type':'.tif', 'input':0, 'truth':1, 'segmt':-1, 'useGPU':False, 'diam':20, 'threshold':-1,'modelName':'untitled'}
        self.explr_btn = Button(win, text = "Browse Directory", command = self.browseFiles)
        self.explr_btn.place(x=100, y=25)
        self.explr_lbl = Label(window, text = '/')
        self.explr_lbl.place(x=250,y=25)

        self.ftype_lbl = Label(win, text='File type')
        self.ftype_lbl.place(x=100, y=50)
        self.ftype_ent = Entry()
        self.ftype_ent.place(x=200, y=50)

        self.input_lbl = Label(win, text='Input Channel')
        self.input_lbl.place(x=100, y=75)
        self.truth_lbl = Label(win, text='Truth Channel')
        self.truth_lbl.place(x=100, y=100)
        self.segmt_lbl = Label(win, text='Segment Channel')
        self.segmt_lbl.place(x=100, y=125)

        self.input_ent = Entry(width = 2)
        self.input_ent.place(x=200, y=75)
        self.truth_ent = Entry(width = 2)
        self.truth_ent.place(x=200, y=100)
        self.segmt_ent = Entry(width = 2)
        self.segmt_ent.place(x=200, y=125)

        self.GPU_lbl = Label(win, text = 'Use GPU')
        self.GPU_lbl.place(x=100, y=150)
        self.ck1 = IntVar()
        self.GPU_chk = Checkbutton(win, variable = self.ck1)
        self.GPU_chk.place(x=200, y=150)

        self.diam_lbl = Label(win, text = 'Estimated Cell Diameter')
        self.diam_lbl.place(x=50, y=175)
        self.diam_ent = Entry(width = 3)
        self.diam_ent.place(x=200, y=175)

        self.thrsh_lbl = Label(win, text = 'Threshold for Positive Label')
        self.thrsh_lbl.place(x=20, y=200)
        self.thrsh_ent = Entry(width = 8)
        self.thrsh_ent.place(x=200, y=200)

        self.model_lbl = Label(win, text = 'Name for trained model?')
        self.model_lbl.place(x=20, y=250)
        self.model_ent = Entry()
        self.model_ent.place(x=200, y=250)

        self.train_btn = Button(win, text='Train Model', command=self.run)
        self.train_btn.place(x=200, y=300)
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
    def run(self):
        self.update_param()
        print(self.settings)
        data_x, data_y = segment.run_pipeline(self.settings['path'], self.settings['type'], self.settings['input'], self.settings['truth'], self.settings['segmt'], self.settings['useGPU'], self.settings['diam'], self.settings['threshold'])
        x = generate_model.init_model(self.settings['diam']*2, self.settings['threshold'])
        generate_model.fit_model(x, data_x, data_y, self.settings['diam']*2, self.settings['modelName'])


window=Tk()
mywin=MyWindow(window)
window.title('Deep Learning Cell Classifier Trainer')
window.geometry("700x400+10+10")
window.mainloop()
