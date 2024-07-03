import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18,ResNet18_Weights
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import urllib.request
import os
import random
#random.seed(0)
#torch.manual_seed(0)
def get_data(txt):
    urls = []
    aim_stack = list("123456789")
    with open(txt, 'r', encoding='utf-8') as f:
        char = f.read(1)
        while char:
            aim_stack.append(char)
            aim_stack.pop(0)
            char = f.read(1)
            if aim_stack == list("drag-img="):
                url = ""
                x = f.read(1)
                while x != "\"":
                    url+=x
                    x = f.read(1)
                urls.append(url)
    return urls
def down_data(aim,path):
    urllib.request.urlretrieve(aim,path)
def run_get():
    urls = get_data("./cats.txt")
    for i in range(len(urls)):
        down_data(urls[i],"./picture/cats/cat%d.jpg"%i)
    urls = get_data("./dogs.txt")
    for i in range(len(urls)):
        down_data(urls[i],"./picture/dogs/dog%d.jpg"%i)
#run_get()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])

class LoadImg(Dataset):
    def __init__(self,img_dir,transform=None):
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths+=[os.path.join(img_dir[i],f) for f in os.listdir(img_dir[i])]
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self,i):
        img_path = self.img_paths[i]
        if(img_path[8] == 'c'):
            label = 1
        else:label = 0
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img,label
class Block(nn.Module):
    def __init__(self,in_planes,planes,stride=1,padding=1):
        super(Block,self).__init__()
        self.layer1 = nn.Conv2d(in_planes,planes,3,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Conv2d(planes,planes,3,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(planes)
        #self.fc = nn.Linear(planes*224*224,2)
        self.downsample = nn.Conv2d(in_planes,planes,1,stride=stride)
    def forward(self,x):
        #identity = x
        out = self.layer1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.bn(out)
        identity = self.downsample(x)
        #print(out.shape, identity.shape)
        out += identity
        out = self.relu(out)
        #out = torch.flatten(out, 1)
        #out = self.fc(out)
        return out
class Model(nn.Module):
    def __init__(self,block):
        super(Model,self).__init__()
        self.planes = 64
        #W = 224 F = 7 ,S = 2 P =3 -> N = (224-7+2*3)/2+1 = 112
        self.conv1 = nn.Conv2d(3, self.planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.planes)
        self.relu = nn.ReLU(inplace=True)
        #W = 112 F = 3 S = 2 P = 1 -> X = (112-3+2*1)/2+1 = 56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block,64)
        self.layer2 = self.make_layer(block,128)
        self.layer3 = self.make_layer(block,256)
        self.layer4 = self.make_layer(block,512)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,2)
    def make_layer(self,block,planes):
        layer = []
        layer.append(block(self.planes,planes))
        self.planes = planes
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
def main():
    def train_Resnet():
        dataset = LoadImg(img_dir=['picture/cats','picture/dogs'],transform=transform)
        dataloader = DataLoader(dataset,batch_size=10,shuffle=True,num_workers=4)
        weights = ResNet18_Weights.DEFAULT
        Resnet18 = resnet18(weights=weights)
        Resnet18.fc = nn.Linear(Resnet18.fc.in_features,2)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(Resnet18.parameters(),lr=0.01,momentum=0.9)
        num_epochs=10
        for epoch in range(num_epochs):
            Resnet18.train()
            for i,(img,label) in enumerate(dataloader):
                #forward
                out = Resnet18(img)
                loss = loss_func(out,label)
                #backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 10 == 0:
                    print("Epoch",epoch + 1,"/",num_epochs,"Step",i + 1,"/",len(dataloader), "Loss:",loss.item())
        torch.save(Resnet18.state_dict(),"resnet18.pth")
    def train_model():
        dataset = LoadImg(img_dir=['picture/cats', 'picture/dogs'], transform=transform)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)
        model = Model(Block)
        model.load_state_dict(torch.load("model.pth"))
        #model = Block(3,8)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        num_epochs = 10
        for epoch in range(num_epochs):
            for i, (img, label) in enumerate(dataloader):
                # forward
                out = model(img)
                loss = loss_func(out, label)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 10 == 0:
                    print("Epoch", epoch + 1, "/", num_epochs, "Step", i + 1, "/", len(dataloader), "Loss:", loss.item())
        torch.save(model.state_dict(), "model2.pth")
    def test():
        test_model = Model(block=Block)
        #test_model = resnet18()
        #test_model.fc = nn.Linear(test_model.fc.in_features, 2)
        test_model.load_state_dict(torch.load("model2.pth"))
        test_model.eval()
        test_img_paths = [os.path.join("test",i) for i in os.listdir("test")]
        def test_(test_img_path):
            test_img = Image.open(test_img_path)
            test_img = transform(test_img).unsqueeze(0)
            with torch.no_grad():
                out = test_model(test_img)
                _,predicted_class = torch.max(out,1)
                if predicted_class.item() == 1:
                    print('Predicted class: cats')
                else:print('Predicted class: dogs')
        for p in test_img_paths:
            test_(p)
    test()
if __name__ == '__main__':
    main()
