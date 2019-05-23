from tkinter import Text

import keras
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Deconvolution2D
from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import UpSampling2D
from keras.layers import Input, Concatenate, Activation,Dropout,ZeroPadding2D,Add,Reshape,LSTM,ConvLSTM2D,TimeDistributed
from keras.models import Model, model_from_json
from keras.optimizers import *
import numpy as np
import json



class DRDE:
    def __init__(self):
        super().__init__()
        self.model = None
        self.batch_size=1
        self.last_best_epoch=0
        self.last_epoch=0

    # Initialize the model
    def init_model(self):
        Input_1 = Input(shape=(100, 150, 1), name='Input_1')
        Input_2 = Input(shape=(100, 150, 1), name='Input_2')
        # 
        #
        # This section will be shown after publishing the paper
        #
        #
        merge_11 = Concatenate(axis=3)([merge_10, Convolution2D_22])
        BatchNormalization_13 = BatchNormalization(name='BatchNormalization_13')(merge_11)
        Activation_13 = Activation(name='Activation_13', activation='relu')(BatchNormalization_13)
        Convolution2D_23 = Convolution2D(name='Convolution2D_23', nb_col=9, nb_filter=1, border_mode='same', nb_row=9)(
            Activation_13)
        Activation_14 = Activation(name='Activation_14', activation='sigmoid')(Convolution2D_23)




        model = Model([Input_1,Input_2], [Activation_14])

        print(model.summary())
        self.model = model
        return model

    # Return the model optimizer
    def get_optimizer(self):
        return RMSprop(lr=1e-3)#Adam()

    # Return the model Loss function
    def get_loss_function(self):
        return 'mean_squared_error'

    # Return the Batch size
    def get_batch_size(self):
        return 1

    # Return the default number of epochs
    def get_num_epoch(self):
        return 100

    # Load model and weights from disk
    def load_model_and_weight(self, model_name):
        # load model
        json_file = open(model_name + '.json', 'r')
        model = json_file.read()
        json_file.close()
        model = model_from_json(model)
        # load weights into model
        model.load_weights(model_name + ".h5")
        print("Loaded model from disk")
        self.model = model
        fp = open(model_name+'_settings.txt', "r")
        data = json.load(fp)
        self.last_epoch = data['last_epoch']
        self.last_best_epoch = data['last_best_epoch']
        print(model.summary())

    # Save model and weights into model directory
    def save_model_and_weight(self, model_name,last_epoch,last_best_epoch):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_name + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_name + '.h5')
        print("Saved model to disk")

        data = {'last_epoch': last_epoch, 'last_best_epoch': last_best_epoch}
        with open(model_name+'_settings.txt', 'w') as fp:
            json.dump(data, fp)

    # Compile the model
    def compile(self):
        self.model.compile(optimizer=self.get_optimizer(), loss=self.get_loss_function(), metrics=['accuracy'])
        return self.model

    # Train the model
    def train(self, x, y, n_epoch=20, batch_size=1):
        self.batch_size=batch_size
        self.model.fit(x, y, epochs=n_epoch, batch_size=batch_size, verbose=1,shuffle=True)

    # Check the error rate on its input test data (x_test & y_test) and print the result in consule
    def get_error_rate(self, x_ts, y_ts,batch_size=8):
        p = self.model.predict(x_ts, batch_size=batch_size, verbose=0)
        mse = np.mean(np.square(y_ts - p))
        print("Error rate is " + str(mse))
        return mse
