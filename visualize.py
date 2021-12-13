import os
import cv2
import math
import time
import torch
from PIL import Image, ImageDraw
from IPython.display import Image as imshow

from utils import *


def plot_boxes (img, boxes, savename = None, class_names = None, color = None):
    colors = torch.FloatTensor ([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);
    def get_color (c, x, max_val):
        ratio = float (x) / max_val * 5
        i = int (math.floor (ratio))
        j = int (math.ceil (ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors [i][c] + ratio * colors [j][c]
        return int (r * 255)
 
    width  = img.width
    height = img.height
    draw = ImageDraw.Draw (img)
    detections = []
    for i in range (len (boxes)):
        box = boxes [i]
        x1 = (box [0] - box [2] / 2.0) * width
        y1 = (box [1] - box [3] / 2.0) * height
        x2 = (box [0] + box [2] / 2.0) * width
        y2 = (box [1] + box [3] / 2.0) * height
        rgb = color if color else (255, 0, 0)
        if len (box) >= 7 and class_names:
            cls_conf = box [5]
            cls_id   = box [6]
            detections += [(cls_conf, class_names [cls_id])]
            classes = len (class_names)
            offset = cls_id * 123457 % classes
            red   = get_color (2, offset, classes)
            green = get_color (1, offset, classes)
            blue  = get_color (0, offset, classes)
            rgb = (red, green, blue)
            draw.rectangle ([x1, y1 - 15, x1 + 6.5 * len (class_names [cls_id]), y1], fill = rgb)
            draw.text ((x1 + 2, y1 - 13), class_names [cls_id], fill = (0, 0, 0))
        draw.rectangle ([x1, y1, x2, y2], outline = rgb, width = 3)
    for (cls_conf, class_name) in sorted (detections, reverse = True):
        print ('%-10s: %f' % (class_name, cls_conf))
    if savename:
        print ('save plot results to %s' %savename)
        img.save (savename)
    return img


def plot_boxes_cv2 (img, boxes, savename = None, class_names = None, color = None):
    colors = torch.FloatTensor ([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);
    def get_color (c, x, max_val):
        ratio = float (x) / max_val * 5
        i = int (math.floor (ratio))
        j = int (math.ceil (ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors [i][c] + ratio * colors [j][c]
        return int (r * 255)
 
    width  = img.shape [1]
    height = img.shape [0]
   
    detections = []
    for i in range (len (boxes)):
        box = boxes [i]
        x1 = int ((box [0] - box [2] / 2.0) * width)
        y1 = int ((box [1] - box [3] / 2.0) * height)
        x2 = int ((box [0] + box [2] / 2.0) * width)
        y2 = int ((box [1] + box [3] / 2.0) * height)
        rgb = (255, 0, 0)
        if len (box) >= 7 and class_names:
            cls_conf = box [5]
            cls_id   = box [6]
            detections += [(cls_conf, class_names [cls_id])]
            classes = len (class_names)
            offset = cls_id * 123457 % classes
            red   = get_color (2, offset, classes)
            green = get_color (1, offset, classes)
            blue  = get_color (0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            cv2.rectangle (img, (x1, y1 - 17), (x1 + 8 * len (class_names [cls_id]) + 10, y1), rgb, -1)
            cv2.putText (img, class_names [cls_id], (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.rectangle (img, (x1, y1), (x2, y2), rgb, 2)
    for (cls_conf, class_name) in sorted (detections, reverse = True):
        print ('%-10s: %f' % (class_name, cls_conf))
    if savename:
        print ('save plot results to %s' %savename)
        cv2.imwrite (savename, img)
    return imshow (savename)


def predict (model, conf_thresh, nms_thresh, img_path, class_names, device, save_to = None):
    assert os.path.exists (img_path), 'Error! Input image does not exists.'
    model.eval ()
    img = Image.open (img_path).convert ('RGB')
 
    tic = time.time ()
    boxes = filtered_boxes (model, device, img.resize ((608, 608)), conf_thresh, nms_thresh)
 
    toc = time.time ()
    print ('Prediction took {:.5f} ms.'.format ((toc - tic) * 1000))
    pred_img = plot_boxes (img, boxes, save_to, class_names)
   
    return pred_img


def predict_cv2 (model, conf_thresh, nms_thresh, img_path, class_names, device, save_to = None):
    assert os.path.exists (img_path), 'Error! Input image does not exists.'
    model.eval ()
    img = cv2.imread (img_path)
 
    tic = time.time ()
    boxes = filtered_boxes (model, device, cv2.resize (img, (608, 608)), conf_thresh, nms_thresh)
 
    toc = time.time ()
    print ('Prediction took {:.5f} ms.'.format ((toc - tic) * 1000))
    pred_img = plot_boxes_cv2 (img, boxes, save_to, class_names)
   
    return pred_img