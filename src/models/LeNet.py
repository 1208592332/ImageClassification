# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 18:26
# @Author  : Spytensor
# @File    : LeNet.py
# @Email   : zhuchaojie@buaa.edu.cn

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet:
    @staticmethod
    def build(config):
        # initialize the model
        model = Sequential()
        inputShape = (config.normal_size,config.normal_size,config.channels)
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(config.classes))
        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        # return the constructed network architecture
        return model