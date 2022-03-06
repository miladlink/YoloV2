import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
from PIL import Image


def gpu2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def find_all_boxes(
    output,
    device,
    conf_thresh,
    num_classes,
    anchors,
    num_anchors,
    only_objectness=1,
    validation=False):
    """ extracting bboxes and confidece from output """
    num_classes, num_anchors = int(num_classes), int(num_anchors)
    anchor_step = int(len(anchors) / num_anchors)
    if output.dim == 3:
       output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)
    
    all_boxes = []
    output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes, batch * num_anchors * h * w)
   
    grid_x = torch.linspace(0, h-1, h).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w).to(device)
    grid_y = torch.linspace(0, w-1, w).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w).to(device)
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y
   
    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).to(device)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).to(device)
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h
 
    det_confs = torch.sigmoid(output[4])
 
    cls_confs = nn.Softmax(dim = 0)(output[5: 5 + num_classes].transpose(0, 1)).data
    cls_max_confs, cls_max_ids = torch.max (cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids   = cls_max_ids.view(-1)
 
    sz_hw  = h * w
    sz_hwa = sz_hw * num_anchors

    det_confs     = det_confs.cpu()
    cls_max_confs = cls_max_confs.cpu()
    cls_max_ids   = gpu2cpu_long(cls_max_ids)
    xs, ys = xs.cpu(), ys.cpu()
    ws, hs = ws.cpu(), hs.cpu()
 
    if validation:
        cls_confs = cls_confs.view(-1, num_classes).cpu()
 
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
 
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw  = ws[ind]
                        bh  = hs[ind]
 
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id   = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)

    return all_boxes


def iou(box1, box2, x1y1x2y2=True):
    """ Intersection Over Union """
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
 
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else: # (x, y, w, h)
        mx = min(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)
        Mx = max(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)
        my = min(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
        My = max(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)
 
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
 
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
   
    corea = 0
    if cw <= 0 or ch <= 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    corea = cw * ch
    uarea = area1 + area2 - corea
    return corea / uarea


def nms(boxes, nms_thresh):
    """ None Max Separetion """
    if len(boxes) == 0:
        return boxes
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]
 
    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
           out_boxes.append(box_i)
           for j in range(i + 1, len(boxes)):
               box_j = boxes[sortIds[j]]
               if iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                  box_j[4] = 0
    return out_boxes


def filtered_boxes(model, device, img, conf_thresh, nms_thresh):
    """ filter best boxes from all boxes """
    model.eval()
   
    if isinstance(img, Image.Image):
        img = transforms.ToTensor ()(img).unsqueeze(0)
    elif type (img) == np.ndarray:
        img = torch.from_numpy(img.transpose (2, 0, 1)).float().div(255.0).unsqueeze(0)
    else:
        print('unknown image type')
        exit(-1)
 
    img = img.to(device)
 
    output = model(img)
    output = output.data
 
    boxes = find_all_boxes(
        output,
        device,
        conf_thresh,
        model.num_classes,
        model.anchors,
        model.num_anchors)[0]
 
    boxes = nms(boxes, nms_thresh)
 
    return boxes