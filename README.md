### 1.本项目为使用图像分类算法对交通标志进行识别
```
注：总结此项目主要为熟悉图像分类的整个流程<br>
    数据处理-->模型选择-->接口封装-->预测

```
### 2.项目结构
```tree
├─data
│  ├─test
│  └─train
└─pythons
        __init__.py  
        main.py 
        model.py
        predict.py
        resnet.py
        utils.py
```
#### 2.1 main.py
基本的参数设置，例如：epoch,learning rate,batch size 等等
#### 2.2 model.py
此处添加的是LeNet-5模型
#### 2.3 predict.py
载入训练好的模型，对新数据进行预测
#### 2.4 resnet.py
构建resnet模型，提供可供选择的：resnet-18,resnet-34,resnet-50,resnet-101,resnet-152，本文件参考了[resnet](https://github.com/raghakot/keras-resnet),仅做了部分修改，主要针对不同的任务做了输入和输出维度的变化。
#### 2.5 utils.py
主要实现数据集的载入，以及训练集、label等的构建，最重要的是keras进行模型训练的参数设置。
