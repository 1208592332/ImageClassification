# -*- coding: utf-8 -*-
# @Time    : 18-6-14 上午9:15
# @Author  : Spytensor
# @File    : main.py
# @Email   : zhuchaojie@buaa.edu.cn

import argparse
import os
from utils import train,load_data

#data path
path_test = "../data/test/"
path_train = "../data/train/"
dirlist_train = os.listdir(path_train)
dirlist_test = os.listdir(path_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",             default = True,                   help = "active this flag to train the model",          action= "store_true")
    parser.add_argument("--predict",           default = False,                  help = "active this flag to predict on new data",      action= "store_true")
    parser.add_argument("--model",             default = "../models/resnet_18_best1.h5",               help = "the path to save trained model")
    parser.add_argument("--epoch",             default = 500,                      help = "number of training epoch",                     type= int)
    parser.add_argument("--lr",                default = 0.001,                  help = "learning rate",                                type= float)
    parser.add_argument("--batch_size",        default = 64,                     help = "bacth size for training",                      type= int)
    parser.add_argument("--norm_size",         default = 64,                    help = "the size of images",                           type= int)
    parser.add_argument("--plot",              default = "../analyse/plot.png",             help = "trained model info"               )
    parser.add_argument("--data_augmentation",              default = True,             help = "data augmentation"               )
    args = parser.parse_args()

    params = {
        "model":args.model,
        "epoch":args.epoch,
        "lr":args.lr,
        "batch_size":args.batch_size,
        "norm_size":args.norm_size,
        "aug":args.data_augmentation,
        "plot":args.plot,
    }
    #loading dataset and generate training data
    train_X,train_y ,train_X_name= load_data(path_train,dirlist_train,params)
    test_X,test_y,test_X_name = load_data(path_test,dirlist_test,params)

    train(train_X,train_y,test_X,test_y,params)





