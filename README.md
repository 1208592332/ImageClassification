>This project aims to help people who know some meachine learning algorithms or deep learning's,but have no idea how to begin a task using what they have learnt.
And in this repository,I have added some CNNs to our models.What you need to do is to change the model name in config.py.

>Networks in our preoject:
    >- LeNet
>- AlexNet  
>- VGGNet
>- ZFNet
>- GoogLeNet
>- ResNet_18/34/50/101/152
>- DenseNet_161
### 1.Project structure
```tree
├─data             
│  ├─test          
│  └─train         
│      ├─00000     
│      ├─00001     
│      ├─00002     
│      ├─00003     
│      ├─00004     
│      ├─00005     
│      ├─00006     
│      ├─00007     
│      ├─00008     
│      └─00009     
├─log              
└─src              
    │  config.py
    │  train.py
    │  utils.py
    │  predict.py
    │
    ├─models
    │  │  AlexNet.py
    │  │  DenseNet.py
    │  │  GooLeNet.py
    │  │  LeNet.py
    │  │  resnet.py
    │  │  VGGNet.py
    │  │  ZFNet.py 
```
### 2.how to use
##### 2.1 for data 
You need to add your deferent category images to the folder "train/",and make a new floder to store your images.For example you have some dog's images ,you can makedir "data/train/dog/",and move your images to it.
##### 2.2 for models
If you want to change the model ,the only thing you need to do is to change the parameter "model_name" in "config.py".
Then do :
>python train.py

##### 2.3 for using the trained model
run:
>python predict.py

### 3. references
[1] Lécun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11):2278-2324.
[2] Krizhevsky A, Sutskever I, Hinton G E. ImageNet classification with deep convolutional neural networks[C]// International Conference on Neural Information Processing Systems. Curran Associates Inc. 2012:1097-1105.
[3] Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. Computer Science, 2014.
[4] Zeiler M D, Fergus R. Visualizing and Understanding Convolutional Networks[J]. 2014, 8689:818-833.
[5] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]// IEEE Conference on Computer Vision and Pattern Recognition. IEEE Computer Society, 2016:770-778.
[6] Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions[C]// IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2015:1-9.
[7] Huang G, Liu Z, Laurens V D M, et al. Densely Connected Convolutional Networks[J]. 2016:2261-2269.
[8] [CNN网络架构演进：从LeNet到DenseNet](https://www.cnblogs.com/skyfsm/p/8451834.html)
[9] [DenseNet-Keras](https://github.com/flyyufelix/DenseNet-Keras)
[10] [keras-resnet](https://github.com/raghakot/keras-resnet)
