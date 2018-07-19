# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 15:51
# @Author  : Spytensor
# @File    : config.py
# @Email   : zhuchaojie@buaa.edu.cn

class DefaultConfigs(object):
	"""docstring for DefaultConfig"""
	train_data_path = "../data/train/"     #训练数据所在路径
	test_data_path = "../data/test/"       #要识别的图像存储路径
	weights_path = "../log/"               #模型保存路径
	normal_size = 64 				        #图像输入网络之前需要被resize的大小
	channels = 3                           #RGB通道数
	epochs = 60                            #训练的epoch次数
	batch_size = 64                         #训练的batch 数
	classes = 10                            #要识别的类数
	data_augmentation = True               #是否使用keras的数据增强模块
	model_name = "ResNet_18"                 #选择所要使用的网络结构名称

config = DefaultConfigs()

#################### 目前支持的网络结构 ####################
					##### AlexNet #####
					##### ResNet_18 #####
					##### ResNet_34 #####
					##### ResNet_50 #####
					##### ResNet_101 #####
					##### ResNet_152 #####
					##### LeNet #####
					##### VGGNet #####
					##### ZFNet #####
					##### GoogLeNet #####
					##### DenseNet_161 #####




