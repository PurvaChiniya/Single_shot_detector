## SSD: Single-Shot MultiBox Detector implementation in Keras

This is a Keras implementation of the SSD model architecture introduced by Wei Liu et al. in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

### Dependencies

* Python 3.x
* Numpy
* Cuda 10.0
* TensorFlow 1.14.0
* Keras 2.0.0
* OpenCV (for data augmentation)
* Beautiful Soup 4.x (to parse XML files)



To train the original SSD300 model on Pascal VOC:

1. Download the datasets:
  ```c
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  ```
2. Download the weights for the convolutionalized VGG-16 or for one of the trained original models provided below.
### Download the convolutionalized VGG-16 weights

In order to train an SSD300 or SSD512 from scratch, download the weights of the fully convolutionalized VGG-16 model trained to convergence on ImageNet classification here:

[`vgg-16_ssd-fcn_ILSVRC-CLS-LOC.h5`](https://drive.google.com/open?id=0B0WbA4IemlxlbFZZaURkMTl2NVU).

This is a modified version of the VGG-16 model from `keras.applications.vgg16`. In particular, the `fc6` and `fc7` layers were convolutionalized and sub-sampled from depth 4096 to 1024, following the paper.
