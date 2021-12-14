# Pytorch YoloV2 implementation from scratch


This repository is simple implementation of `YOLOv2 algorithm` for better understanding and use it for more **object detection** usage. This project based on Pytorch. The code of project is so easy and clear.

## Dataset

Pretrained weights in this implemetation are based on training yolo team on COCO trainval dataset

## Usage

**You can have your own object detection machine**

**Note**: The tests of this repo run with `cpu` mode so if you use `gpu` prediction become much faster

### Clone the repository

```bash
git clone https://github.com/miladlink/YoloV2.git

cd YoloV2
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Pretrained Weights

* Download used weights in this project from [here](https://pjreddie.com/media/files/yolov2.weights) or go to weights and click to yolov2.weights

**or**

```bash
chmod +x weights/get_weights.sh
weights/get_weights.sh
```

### Help

```bash
python detect.py -h
```

![image](https://user-images.githubusercontent.com/81680367/145953199-0addc1c0-d63d-4462-890d-10f6a9a8c8e4.png)

### Detection

```bash
python detect.py -w path/to/weights\
-ct <conf_thresh>\
-nt <nms_thresh>\
-p path/to/img\
-s
```

## Some Result image

you can see some examples in `yolov2_examples.ipynb`

![dog](https://user-images.githubusercontent.com/81680367/145892095-2e804947-7fd7-436b-b907-f08f14e3b6e6.jpg)

![person](https://user-images.githubusercontent.com/81680367/146009270-38b256f2-5b66-491d-933c-6a92eceb71f9.jpg)

![giraffe](https://user-images.githubusercontent.com/81680367/145892114-45386e1f-5923-40b0-a6c7-7d330fe1d099.jpg)

**Note**: TinyYoloV2 trained by pascal VOC can not predict above image

for more information about TinyYoloV2 implementations you can visit

https://github.com/miladlink/TinyYoloV2