# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 16:17
# @Author  : Spytensor
# @File    : train.py
# @Email   : zhuchaojie@buaa.edu.cn

from utils import build_model,load_data
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from config import config

def train(config,train_x,train_y,dev_x,dev_y):
    model = build_model(config)
    lr_reducer = ReduceLROnPlateau(factor=0.005, cooldown=0, patience=5, min_lr=0.5e-6,verbose=1)      #设置学习率衰减
    early_stopper = EarlyStopping(min_delta=0.001, patience=10,verbose=1)                                     #设置早停参数
    checkpoint = ModelCheckpoint(config.weights_path + config.model_name + "_model.h5",
                                 monitor="val_acc", verbose=1,
                                 save_best_only=True, save_weights_only=True,mode="max")            #保存训练过程中，在验证集上效果最好的模型
    #使用数据增强
    if config.data_augmentation:
        print("using data augmentation method")
        data_aug = ImageDataGenerator(
            rotation_range=90,              #图像旋转的角度
            width_shift_range=0.2,          #左右平移参数
            height_shift_range=0.2,         #上下平移参数
            zoom_range=0.3,                 #随机放大或者缩小
            horizontal_flip=True,           #随机翻转
        )
        data_aug.fit(train_x)
        model.fit_generator(
            data_aug.flow(train_x,train_y,batch_size=config.batch_size),
            steps_per_epoch=train_x.shape[0] // config.batch_size,
            validation_data=(dev_x,dev_y),
            shuffle=True,
            epochs=config.epochs,verbose=1,max_queue_size=100,
            callbacks=[lr_reducer,early_stopper,checkpoint]
        )
    else:
        print("don't use data augmentation method")
        model.fit(train_x,train_y,batch_size = config.batch_size,
                  nb_epoch=config.epochs,
                  validation_data=(dev_x, dev_y),
                  shuffle=True,
                  callbacks=[lr_reducer, early_stopper, checkpoint]
                  )

if __name__ == "__main__":
    images_data, labels = load_data(config)
    train_x,dev_x,train_y,dev_y = train_test_split(images_data,labels,test_size=0.25) #随机切分数据集，分为训练和验证集
    train(config,train_x,train_y,dev_x,dev_y)
