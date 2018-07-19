# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 19:26
# @Author  : Spytensor
# @File    : predict.py
# @Email   : zhuchaojie@buaa.edu.cn

import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from config import config
from utils import build_model
def predict(config):
    weights_path = config.weights_path + config.model_name + "_model.h5"
    data_path = config.test_data_path
    data_list = os.listdir(data_path)
    data = []
    for file in data_list:
        file_name = data_path + file
        image = cv2.imread(file_name)
        image = cv2.resize(image, (config.normal_size,config.normal_size))
        image = img_to_array(image)
        data.append(image)
    data = np.array(data, dtype="float") / 255.0

    model = build_model(config)
    model.load_weights(weights_path)
    pred = model.predict(data)
    print(pred)
predict(config)