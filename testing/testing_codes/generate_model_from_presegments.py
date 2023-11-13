import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers, initializers, callbacks
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
import tensorflow as tf
import keras_tuner as kt

import os
import matplotlib.pyplot as plt
import pathlib
from skimage import io
from skimage.transform import resize
import torch
import cv2



bf_imgs = []
for i in range(7974):
    filenames = str(i) + '_bf.tif'
    bf_imgs.append(io.imread(filenames))
input_size = bf_imgs[0].shape[0]
labels = np.loadtxt('labels.txt',delimiter = ',')
print(labels)

bin_label = np.where(labels > 1000, 1, 0)

def init_model(input_size, threshold):
    if threshold == -1:
        loss_type = 'poisson'
        metric_type = 'mean_squared_error'
        fin_activation = 'relu'
    else:
        loss_type = 'binary_crossentropy'
        metric_type = 'accuracy'
        fin_activation = 'sigmoid'

    initializer = initializers.RandomNormal(mean=0., stddev=1.)

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_size, input_size, 1)))
    model.add(layers.Conv2D(64,  kernel_size = (3,3), strides = (2,2), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, kernel_size = (5,5), strides = (2,2), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, kernel_initializer=initializer, bias_initializer=initializer, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2048, kernel_initializer=initializer, bias_initializer=initializer, activation = 'relu'))
    model.add(layers.Dense(256, kernel_initializer=initializer, bias_initializer=initializer, activation = 'relu'))
    model.add(layers.Dense(1, activation = fin_activation))
    opt = optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer = opt, loss = loss_type, metrics = [metric_type])

    return model

def init_model2(hyp):
    # if threshold == -1:
    #     loss_type = 'poisson'
    #     metric_type = 'mean_squared_error'
    #     fin_activation = 'relu'
    # else:
    #     loss_type = 'binary_crossentropy'
    #     metric_type = 'accuracy'
    #     fin_activation = 'sigmoid'
    loss_type = 'binary_crossentropy'
    metric_type = 'accuracy'
    fin_activation = 'sigmoid'

    initializer = initializers.RandomNormal(mean=0., stddev=1.)

    droprate = hyp.Choice(name = 'dropoutRate', values = np.arange(0.0, 0.5, 0.05).tolist())
    neuron_1 = hyp.Int(name = 'first_depth', min_value = 256, max_value = 4096, step = 256)
    neuron_2 = hyp.Int(name = 'secnd_depth', min_value = 256, max_value = 2048, step = 256)

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_size, input_size, 1)))
    model.add(layers.Conv2D(64,  kernel_size = (3,3), strides = (2,2), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, kernel_size = (5,5), strides = (2,2), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(neuron_1, kernel_initializer=initializer, bias_initializer=initializer, activation = 'relu'))
    model.add(layers.Dropout(droprate))
    model.add(layers.Dense(neuron_2, kernel_initializer=initializer, bias_initializer=initializer, activation = 'relu'))
    model.add(layers.Dense(256, kernel_initializer=initializer, bias_initializer=initializer, activation = 'relu'))
    model.add(layers.Dense(1, activation = fin_activation))
    opt = optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer = opt, loss = loss_type, metrics = [metric_type])

    return model

def fit_model(model, data_x, data_y, input_size, modelName):
    x,y = data_org(data_x, data_y, input_size)
    best_callback = [callbacks.ModelCheckpoint(filepath= modelName + '.h5', monitor='val_accuracy', save_best_only = True, mode='max')]

    history = model.fit(x, y, batch_size = 16, epochs = 100, validation_split=0.2, shuffle=True, callbacks = best_callback)

    print('Model has been saved in the supplied directory as ' + modelName + '.h5')
    return history

def fit_model2(model, data_x, data_y, input_size, thres, modelName):
    tuner = kt.Hyperband(model, objective='val_accuracy', max_epochs = 10, directory='', project_name='bestmodelkt')
    stop_early = [callbacks.EarlyStopping(monitor='val_accuracy', patience = 10)]

    x,y = data_org(data_x, data_y, input_size)

    tuner.search(x, y, epochs = 25, validation_split=0.5, callbacks = stop_early)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print('Model has been optimized')

    hypermodel = tuner.hypermodel.build(best_hps)
    history = hypermodel.fit(x,y,epochs=100,validation_split=0.5)


def data_org(data_x, data_y, size_selection):
    data_x = np.stack(data_x,axis=0)
    x = data_x.reshape(-1, size_selection, size_selection, 1)
    return x,data_y

# model = init_model(input_size, 1000)
fit_model2(init_model2, bf_imgs, bin_label, input_size, 100,'test')



def plot_train_curves(history):
  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  epochs = range(1, len(accuracy) + 1)

  plt.plot(epochs, accuracy, "b--", label="Training accuracy")
  plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
  plt.title("Training and Validation Accuracy")
  plt.legend()
  plt.figure()

  plt.plot(epochs, loss, "b--", label = "Training loss")
  plt.plot(epochs,val_loss, "b", label = "Validation loss")
  plt.title("Training and Validation Loss")
  plt.legend()
  plt.show()

plot_train_curves(history)
