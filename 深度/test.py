import torch.nn as nn
from torch.utils.data import dataloader
from torch.utils.data import dataset
from torchvision import transforms
from PIL import Image
import os

class LoadImg():
    def __init__(self,dir,transforms):
        self.paths = []
        for i in range(len(dir)):
            self.paths+=[os.path.join(dir[i])+f for f in os.listdir(dir[i])]
        self.transform = transforms
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, item):
        img_path = self.paths[item]
        img = Image.open(img_path)
        label = []
        if self.transform:
            img = self.transform(img)
        return label,img
for i ,j in zip([1,2,3,4],['a','b','c','d']):
    print(i)