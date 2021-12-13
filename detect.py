import torch
from init import config
from model import load_model
from visualize import predict

import argparse

parser = argparse.ArgumentParser (description = 'YoloV2 Detection')

parser.add_argument('-w', '--weight_path', type = str, metavar = '', default = 'weights/yolov2.weights', help = 'weight path')
parser.add_argument('-ct', '--conf_thresh', type = float, metavar='', default = 0.4, help = 'confidence threshold')
parser.add_argument('-nt', '--nms_thresh', type = float, metavar='', default = 0.4, help = 'non max separetion threshold')
parser.add_argument('-p', '--img_path', type = str, metavar = '', default = 'images/dog.jpg', help = 'image path')

save = parser.add_mutually_exclusive_group()
save.add_argument ('-s', '--save',  action = 'store_true', help = 'save for predictions')

args = parser.parse_args ()

if __name__ == '__main__':
    # load model
    model = load_model (args.weight_path, config.device)
    # if save output wanted
    if args.save:
        save_path = 'weights/output.jpg'

    # detect bounding box and draw it
    predict (
        model, 
        args.conf_thresh, 
        args.nms_thresh, 
        args.img_path, 
        config.class_names, 
        config.device, 
        save_path)

