# VGGFace2 Dataset for Face Recognition ([website](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/))


The dataset contains 3.31 million images of 9131 subjects (identities), with an average of 362.6 images for each subject. Images are downloaded from Google Image Search and have large variations in pose, age, illumination, ethnicity and profession (e.g. actors, athletes, politicians). The whole dataset is split to a training set (including **8631** identites) and a test set (including **500** identites).
```
VGGFace2: A dataset for recognising faces across pose and age, 
Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman,
In FG 2018. 
```
<div align="center">
  <img src="https://github.com/ox-vgg/vgg_face2/blob/master/web_page_img.png">
</div>

## News
| Date     | Update |
|----------|--------|
|2018-10-01 | Models imported to PyTorch. Example scripts for cropping faces and evaluating on IJB-B can be found in the folder 'standard_evaluation'. 
|2018-09-18 | Models with lower-dimensional embedding layers for feature representation.
|2018-04-10 | New models trained on VGGFace2 (see below). Training details can be found in the [paper](http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf). |
## Trained models
We report 1:1 verification performance (center 224x224 crop from resized image with shorter side = 256) on IJB-B [1] for reference ([ROC](https://github.com/lishen-shirley/vggface2/blob/master/ijbb_roc.png). Higher is better). More evaluation results can be found in the [paper](http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf). Models in *pretrain* setting are trained on MS-Celeb-1M [2] dataset and then fine-tuned on VGGFace2 dataset. ResNet-50 models follow the architectural configuration in [3] and SE-ResNet-50 models follow the one in [4]. "<model-#D>" means that a lower-dimensional embedding layer is stacked on the top of the original final feature layer adjacent to classifier. The models are trained with standard softmax loss.

**Python Version**: To load the PyTorch model, make sure you are using python2.x. 

**TF Model(Keras)**: Please check this repo([https://github.com/WeidiXie/Keras-VGGFace2-ResNet50]). 

| Architecture   | Feat dim | Pretrain | TAR@FAR = 0.001 | TAR@FAR = 0.01 | Model Link |
|:-:|:-:|:-:|:-:|:-:|:-:|
|   ResNet-50    | 2048 |  N  | 0.878 | 0.938 | [Caffe](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/caffe/resnet50_scratch_caffe.tar.gz), [MatConvNet](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/matconvnet/resnet50_scratch_mat.tar.gz), [TF], [PyTorch](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/resnet50_scratch_pytorch.tar.gz) |
|   ResNet-50    | 2048 |  Y  | 0.891 | 0.947 | [Caffe](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/caffe/resnet50_ft_caffe.tar.gz), [MatConvNet](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/matconvnet/resnet50_ft_mat.tar.gz), [TF], [PyTorch](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/resnet50_ft_pytorch.tar.gz) |
| SE-ResNet-50   | 2048 |  N  | 0.888 | 0.949 | [Caffe](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/caffe/senet50_scratch_caffe.tar.gz), [MatConvNet](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/matconvnet/senet50_scratch_mat.tar.gz), [TF], [PyTorch](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/senet50_scratch_pytorch.tar.gz)|
| SE-ResNet-50   | 2048 |  Y  | 0.908 | 0.956 | [Caffe](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/caffe/senet50_ft_caffe.tar.gz), [MatConvNet](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/matconvnet/senet50_ft_mat.tar.gz), [TF], [PyTorch](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/senet50_ft_pytorch.tar.gz)|
| ResNet-50-256D | 256  |  Y  | 0.898 | 0.956 | [Caffe](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/caffe/resnet50_256_caffe.tar.gz), [MatConvNet](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/matconvnet/resnet50_256_mat.tar.gz), [TF], [PyTorch](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/resnet50_256_pytorch.tar.gz) |
| ResNet-50-128D | 128  |  Y  | 0.904 | 0.956 | [Caffe](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/caffe/resnet50_128_caffe.tar.gz), [MatConvNet](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/matconvnet/resnet50_128_mat.tar.gz), [TF], [PyTorch](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/resnet50_128_pytorch.tar.gz) |
| SE-ResNet-50-256D|  256    |    Y   | 0.912 | 0.965 | [Caffe](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/caffe/senet50_256_caffe.tar.gz), [MatConvNet](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/matconvnet/senet50_256_mat.tar.gz), [TF], [PyTorch](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/senet50_256_pytorch.tar.gz)|
| SE-ResNet-50-128D|  128    |    Y   | 0.910 | 0.959 | [Caffe](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/caffe/senet50_128_caffe.tar.gz), [MatConvNet](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/matconvnet/senet50_128_mat.tar.gz), [TF], [PyTorch](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/senet50_128_pytorch.tar.gz) |
## Compatibility

**Caffe**: SE models use the "Axpy" layer which is a combination of two consecutive operations *channel-wise scale* and *element-wise summation* (More information can be found [here](https://github.com/hujie-frank/SENet).) Please note that the input mean vector is in BGR order as opencv is used for loading images.

**MatConvNet**: This code uses the following three modules: 
* [autonn](https://github.com/vlfeat/autonn) - a wrapper for MatConvNet
* [mcnExtraLayers](https://github.com/albanie/mcnExtraLayers) - some useful additional layers
* [mcnSENets](https://github.com/albanie/mcnSENets) - supporting layers for SE models

All of these can be setup directly with `vl_contrib` (i.e. run `vl_contrib install <module-name>` then `vl_contrib setup 
<module-name>`).

**TensorFlow**: coming soon.

**PyTorch**: Imported from Caffe models using the tool [5]. Models can be loaded by using:

```
import imp
import torch
# 'senet50_256_pytorch' is the model name
MainModel = imp.load_source('MainModel', 'senet50_256_pytorch.py') 
model = torch.load('senet50_256_pytorch.pth')
```

## Pre-processing:
We use [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment) for face detection. This bounding box is then extended by a factor 0.3 (except the extension outside image) to include the whole head, which is used as network input (Please note that the released faces are based on a larger extension ratio 1.0. The coordinates of bounding boxes and 5 facial keypoints referring to the loosely cropped faces can be found [here](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/).
## Reference
[1] C. Whitelam, E. Taborsky, A. Blanton, B. Maze, J. Adams, T. Miller, N. Kalka, A. K. Jain, J. A. Duncan, K. Allen, J. Cheney and P. Grother. IARIA Janus Benchmark-B Face Dataset. In CVPR Workshop on Biometrics 2017.

[2] Y. Guo, L. Zhang, Y. Hu, X. He, and J. Gao. MS-Celeb-1M: A Dataset and Benchmark for Large Scale Face Recognition. In ECCV 2016.

[3] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR 2016.

[4] J. Hu, L. Shen and G. Sun. Squeeze-and-Excitation Networks. In CVPR 2018.

[5] C. Chen, J. Yao, R. Zhang, Y. Zhou, T. Qin, T. Zhan and Q. Wang. MMdnn:http://github.com/Microsoft/MMdnn 

## Acknowledgements
This research is based upon work supported by the Office of the Director of National Intelligence (ODNI), Intel- ligence Advanced Research Projects Activity (IARPA), via contract number 2014-14071600010. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purpose notwithstanding any copyright annotation thereon.

We would like to thank Samuel Albanie for his help in model converting.
