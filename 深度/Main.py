import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.models import vgg16
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import urllib.request
import matplotlib.pyplot as plt
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
    transforms.CenterCrop(224),
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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by reducing the variance of the residual branches' outputs at initialization.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
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

    def forward(self, x):
        return self._forward_impl(x)


def resnet18(num_classes=1000, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def main():
    def train():
        dataset = LoadImg(img_dir=['picture/cats','picture/dogs'],transform=transform)
        dataloader = DataLoader(dataset,batch_size=10,shuffle=True,num_workers=4)
        Resnet18 = resnet18()
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
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        torch.save(Resnet18.state_dict(),"resnet18.pth")
    def test():
        test_model = resnet18()
        test_model.fc = nn.Linear(test_model.fc.in_features, 2)
        test_model.load_state_dict(torch.load("resnet18.pth"))
        test_model.eval()
        test_img_paths = ['picture/cats/cat0.jpg','picture/cats/cat23.jpg','picture/dogs/dog23.jpg']
        def test_(test_img_path):
            test_img = Image.open(test_img_path)
            test_img = transform(test_img).unsqueeze(0)
            with torch.no_grad():
                out = test_model(test_img)
                _,predicted_class = torch.max(out,1)
                if predicted_class == 1:
                    print(f'Predicted class: cats')
                else:print(f'Predicted class: dogs')
        for p in test_img_paths:
            test_(p)
    test()
if __name__ == '__main__':
    main()