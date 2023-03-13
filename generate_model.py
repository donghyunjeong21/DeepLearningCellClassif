import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
import tensorflow as tf

def init_model(input_size, threshold):
    if threshold == -1:
        loss_type = 'poisson'
        metric_type = 'mean_squared_error'
        fin_activation = 'relu'
    else:
        loss_type = 'binary_crossentropy'
        metric_type = 'accuracy'
        fin_activation = 'softmax'
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_size, input_size, 1)))
    model.add(layers.Conv2D(128, kernel_size = (2,2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size = 2))
    model.add(layers.Conv2D(64, kernel_size = (4,4), activation='relu'))
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(16, activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation = fin_activation))
    opt = optimizers.Adam(learning_rate=0.0000001)
    model.compile(optimizer = opt, loss = loss_type, metrics = [metric_type])

    return model

def fit_model(model, data_x, data_y, input_size, modelName):
    x,y = data_org(data_x, data_y, input_size)
    history = model.fit(x, y, batch_size = 64, epochs = 50, shuffle = True, validation_split=0.2)
    model.save(modelName + '.h5')
    print('Model has been saved in the supplied directory as ' + modelName + '.h5')

def data_org(data_x, data_y, size_selection):
    data_x = np.stack(data_x,axis=0)
    x = data_x.reshape(-1, size_selection, size_selection, 1)
    return x,data_y
