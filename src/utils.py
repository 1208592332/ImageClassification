# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 15:52
# @Author  : Spytensor
# @File    : utils.py
# @Email   : zhuchaojie@buaa.edu.cn

from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from models import AlexNet,resnet
from models.LeNet import LeNet
from models.VGGNet import VGG_16
from models.ZFNet import ZF_Net
from models.GooLeNet import GoogLeNet
from models.DenseNet import DenseNet_161
import cv2
import os
import numpy as np

np.random.seed(42)

def load_data(config):
    labels = []                                                         #存放标签信息
    images_data = []                                                    #存放图片信息
    print("loading dataset......")
    data_path = config.	train_data_path                                  #也即是train文件夹
    category_paths = os.listdir(data_path)                              #每个类别所在的文件夹，返回list格式
    category_paths = list(map(lambda x:data_path+x,category_paths))     #组合成合法的路径，如../data/train/00000
    np.random.shuffle(category_paths)
    for category_path in category_paths:
        images_files_list = os.listdir(category_path)                   #获取每个类别下的图像名称
        print(category_path)
        for image_file in images_files_list:
            file_name = category_path + "/"+image_file                  #每张图片的路径，便于读取
            label = int(category_path[-2:])                                  #提取类别信息
            labels.append(label)

            image = cv2.imread(file_name)                               #使用opencv读取图像
            image = cv2.resize(image,(config.normal_size,config.normal_size))
            image = img_to_array(image)                                 #将图像转换成array形式
            images_data.append(image)

    #缩放图像数据
    images_data = np.array(images_data,dtype="float") / 255.0
    labels = np.array(labels)                                           #将label转换成np.array格式
    labels = to_categorical(labels, num_classes=config.classes)

    return images_data,labels

def build_model(config):
    #根据选择的网络模型构建
    if config.model_name == "AlexNet":
        model = AlexNet.AlexNet(config)
    elif config.model_name == "ResNet_18":
        model = resnet.ResnetBuilder.build_resnet_18(config)
    elif config.model_name == "ResNet_34":
        model = resnet.ResnetBuilder.build_resnet_34(config)
    elif config.model_name == "ResNet_50":
        model = resnet.ResnetBuilder.build_resnet_50(config)
    elif config.model_name == "ResNet_101":
        model = resnet.ResnetBuilder.build_resnet_101(config)
    elif config.model_name == "ResNet_152":
        model = resnet.ResnetBuilder.build_resnet_152(config)
    elif config.model_name == "LeNet":
        model = LeNet().build(config)
    elif config.model_name == "VGGNet":
        model = VGG_16(config)
    elif config.model_name == "ZFNet":
        model = ZF_Net(config)
    elif config.model_name == "GoogLeNet":
        model = GoogLeNet(config)
    elif config.model_name == "DenseNet_161":
        model = DenseNet_161(config)
    else:
        print("The model you have selected doesn't exists!")
    return model




