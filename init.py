import torch
from model import load_model

with open ('class_names', 'r') as f:
    class_names = f.read ().split ('\n')

device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')

model = load_model ('weights/yolov2.weights', device)

class config:
    def __init__ (self):
        self.device = device
        self.class_names = class_names
        self.model = load_model ('weights/yolov2.weights', device)