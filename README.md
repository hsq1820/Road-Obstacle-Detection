[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Obstacle Detection With StixelNet #

## Dependencies ##
- tested on the following environment:
  + OS: tested on windows 11
  + Tensorflow 2.0.0
  + python 3.7
- installing the dependencies:
  + python3 -m pip install -r requirements.txt

## Training Data  ##
### Kitti Raw Dataset ###

### [Ground Truth](https://sites.google.com/view/danlevi/datasets)
### Downloading the customized dataset for this repository ###

```bash
    python3 ./scripts/download_kitti_stixels.py
```
*the dataset is about 5.4G, so would take sometime until finishing downloading.*

## StixelNet Model ##
***

![StixelNet](./docs/images/network.png)

## Training ##
After downloading the dataset, run
```bash
    python3 ./train.py
```
model weights will be saved into ./saved_models directory

## Test one image ##
***

- Download pretrained model weights with
```bash
https://drive.google.com/uc?id=1xbn6O4GpQ2CjRkh-i-7eNfHDktA06hwY
```

- Test on an image
```bash
    python3 ./test-dataset-image.py
```

## References ##
- [StixelNet: A Deep Convolutional Network for Obstacle Detection and Road Segmentation](http://www.bmva.org/bmvc/2015/papers/paper109/paper109.pdf)
- [Real-time category-based and general obstacle detection for autonomous driving](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w3/Garnett_Real-Time_Category-Based_and_ICCV_2017_paper.pdf)
