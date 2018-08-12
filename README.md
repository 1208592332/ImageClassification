
This project aims to help people who know some meachine learning algorithms or deep learning's,but have no idea how to begin a task using what they have learnt.
And in this repository,I have added some CNNs to our models.What you need to do is to change the model name in config.py.

>Networks in our preoject:
>- LeNet
>- AlexNet  
>- VGGNet
>- ZFNet
>- GoogLeNet
>- ResNet_18/34/50/101/152
>- DenseNet_161

#0.Requirements
keras >=2.1.5
tensorflow >=1.8
opencv-python >= 3.4.0.12
# 1.Project structure
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
# 2.how to use
## 2.1 for data 
You need to add your deferent category images to the folder "train/",and make a new floder to store your images.For example you have some dog's images ,you can makedir "data/train/dog/",and move your images to it.
## 2.2 for models
If you want to change the model ,the only thing you need to do is to change the parameter "model_name" in "config.py".
Then do :
>python train.py

## 2.3 for using the trained model
run:
>python predict.py

# 3. references
[Lécun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11):2278-2324.](http://www.cs.princeton.edu/courses/archive/spr08/cos598B/Lectures/LeCunEtAl.pdf)<br>
[Krizhevsky A, Sutskever I, Hinton G E. ImageNet classification with deep convolutional neural networks[C]// International Conference on Neural Information Processing Systems. Curran Associates Inc. 2012:1097-1105.](http://ml.informatik.uni-freiburg.de/former/_media/teaching/ws1314/dl/talk_simon_group2.pdf)<br>
 [Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. Computer Science, 2014.](https://arxiv.org/pdf/1409.1556.pdf)<br>
 [Zeiler M D, Fergus R. Visualizing and Understanding Convolutional Networks[J]. 2014, 8689:818-833.](https://arxiv.org/pdf/1311.2901.pdf)<br>
 [He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]// IEEE Conference on Computer Vision and Pattern Recognition. IEEE Computer Society, 2016:770-778.](https://arxiv.org/pdf/1512.03385)<br>
 [Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions[C]// IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2015:1-9.](https://arxiv.org/pdf/1409.4842.pdf)<br>
[Huang G, Liu Z, Laurens V D M, et al. Densely Connected Convolutional Networks[J]. 2016:2261-2269.](https://arxiv.org/pdf/1608.06993.pdf)<br>
 [Introduce the cnns from LeNet to DensNet](https://www.cnblogs.com/skyfsm/p/8451834.html)<br>
 [DenseNet-Keras](https://github.com/flyyufelix/DenseNet-Keras)<br>
[keras-resnet](https://github.com/raghakot/keras-resnet)<br>
$$\begin{cases} 
		a_{11}x_1&+&a_{12}x_2&+&\cdots&+a_{1n}x_n&=&b_1\\
		&&&&\vdots\\
		a_{n1}x_1&+&a_{n2}x_2&+&\cdots&+a_{nn}x_n&=&b_n&			
	\end{cases}$$
