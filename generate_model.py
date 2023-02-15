import numpy as np
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

def init_model(input_size, threshold):
    if threshold == -1:
        loss_type = 'poisson'
        metric_type = 'mean_squared_error'
        fin_activation = 'relu'
    else:
        loss_type = 'binary_crossentropy'
        metric_type = 'accuracy'
        fin_activation = 'sigmoid'
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_size, input_size, 1)))
    model.add(layers.Conv2D(128, kernel_size = (2,2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size = 2))
    model.add(layers.Conv2D(64, kernel_size = (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size = 2))
    model.add(layers.Conv2D(32, kernel_size = (3,3), activation='relu'))
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dense(64, activation = 'relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation = 'relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation = 'relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation = fin_activation))

    model.compile(optimizer = 'adam', loss = loss_type, metrics = [metric_type])

    return model

def fit_model(model, data_x, data_y, input_size, modelName):
    train_x, test_x, train_y, test_y = data_org(data_x, data_y, input_size)
    history = model.fit(train_x, train_y, batch_size = 100, epochs = 20, shuffle = True, validation_data = (test_x, test_y))
    model.save(modelName + '.h5')
    print('Model has been saved in the supplied directory as ' + modelName + '.h5')

def data_org(data_x, data_y, size_selection):
    data_x = np.stack(data_x,axis=0)
    train_x, test_x, train_y, test_y = train_test_split(data_x,data_y,test_size = 0.2)
    train_x = train_x.reshape(-1, size_selection, size_selection, 1)
    test_x = test_x.reshape(-1, size_selection, size_selection, 1)
    return train_x, test_x, train_y, test_y
