# -*- coding: utf-8 -*-
# @Time    : 18-6-14 上午9:15
# @Author  : Spytensor
# @File    : utils.py
# @Email   : zhuchaojie@buaa.edu.cn

# import the necessary packages
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
from model import LeNet
import resnet
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tensorflow as tf

#1. setting the GPU device info
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 

#2. loading data
def load_data(path,dir_list,params):
    print("[INFO] loading images...")
    data = []
    labels = []
    file_details_name = []
    # grab the image paths and randomly shuffle them
    data_list = list(map(lambda x: path + x, dir_list))
    for data_dir in data_list:
        print("loading data : ",data_dir)
        files_list = os.listdir(data_dir)
        #generate the labels
        for file in files_list:
            if data_dir.split("/")[-1][:2] == "ng":
                label = 0
            else:
                label = 1
            name = data_dir+"/"+file
            labels.append(label)            #label
            image = cv2.imread(name)
            image = cv2.resize(image, (params["norm_size"], params["norm_size"]))
            image = img_to_array(image)
            data.append(image)
            file_details_name.append(name)
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=2)
    return data, labels,file_details_name

#3. training info
def train(trainX, trainY, testX, testY,params):
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('../analyse/resnet18_audioTest.csv')
    mean_image = np.mean(trainX, axis=0)
    trainX -= mean_image
    testX -= mean_image
    trainX /= 128.
    testX /= 128.
    

    print("[INFO] compiling model...")
    model = resnet.ResnetBuilder.build_resnet_50((3, params["norm_size"], params["norm_size"]), 2)
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    # using data augmentation or not
    if not params["aug"]:
        print("Not using data augmentation.")
        checkpoint = ModelCheckpoint(params["model"],monitor="val_acc",verbose=1,save_best_only=True,mode="max")

        model.fit(trainX,trainY,
              batch_size=params["batch_size"],
              nb_epoch=params["epoch"],
              validation_data=(testX, testY),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger,checkpoint])
    else:
        print("Using real-time data augmentation.")
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(trainX)
        
        #store the best model
        checkpoint = ModelCheckpoint(params["model"],monitor="val_acc",verbose=1,save_best_only=True,mode="max")


        H = model.fit_generator(datagen.flow(trainX, trainY, batch_size=params["batch_size"]),
                        steps_per_epoch=trainX.shape[0] // params["batch_size"],
                        validation_data=(testX, testY),
                        epochs=params["epoch"], verbose=1, max_q_size=100,
                        callbacks=[lr_reducer, early_stopper, csv_logger,checkpoint])
         
    # save the model to disk
    print("[INFO] saving last model...")
    #model.save(params["model"])




