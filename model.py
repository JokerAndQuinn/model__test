import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt



class model1(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(model1,self).__init__()
        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)



    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        #print(out.shape)
        out=F.relu(self.bn2(self.conv2(out)))
        return out

class model2(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(model2,self).__init__()
        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.conv3 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(ch_out)


    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        return out

class model3(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(model3,self).__init__()
        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=5,stride=1,padding=2)
        self.bn1=nn.BatchNorm2d(ch_out)



    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))

        return out

class model4(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(model4,self).__init__()
        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)
        self.bn1=nn.BatchNorm2d(ch_out)



    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))

        return out

class model(nn.Module):

    def __init__(self,):
        super(model,self).__init__()


        self.module01 = model1(3, 64, stride=1)
        self.pool01 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.module02 = model1(64, 128, stride=1)
        self.pool02 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.module03 = model2(128, 256, stride=1)
        self.pool03 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.module04 = model2(256, 512, stride=1)
        self.pool04 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.module05 = model2(512,512,stride=1)


        self.module_1=model3(3,24,stride=1)
        self.pool_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.module_2=model3(24,24,stride=1)
        self.pool_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.module_3=model3(24,24,stride=1)
        self.pool_3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.module=model4(512+24,1,stride=1)

        self.outlayer=nn.Linear(28*28,506)
        # self.outlayer=nn.Linear(128*128,510)



    def forward(self,x):
        y1 = self.module01(x)
        y1 = self.pool01(y1)
        y1 = self.module02(y1)
        y1 = self.pool02(y1)
        y1 = self.module03(y1)
        y1 = self.pool03(y1)
        y1 = self.module04(y1)
        y1 = self.pool04(y1)
        y1 = self.module05(y1)


        y2 = self.module_1(x)
        y2 = self.pool_1(y2)
        y2 = self.module_2(y2)
        y2 = self.pool_2(y2)
        y2 = self.module_3(y2)
        y2 = self.pool_3(y2)
        # print(y1.shape,y2.shape)
        y = torch.cat([y1,y2],dim=1)
        # print(y.shape)
        y = self.module(y)
        # print(y.shape)


        y=y.view(y.size(0),-1)
        out=self.outlayer(y)

        return out


def main():
    module1 = model1(3, 64)
    tmp=torch.randn(2,3,224,224)
    out=module1(tmp)
    print('model1:',out.shape)

    module2 = model2(3, 64)
    tmp = torch.randn(2, 3, 224, 224)
    out = module2(tmp)
    print('model2:', out.shape)

    module3 = model3(3, 64)
    tmp = torch.randn(2, 3, 224, 224)
    out = module3(tmp)
    print('model3:', out.shape)

    module4 = model4(3, 64)
    tmp = torch.randn(2, 3, 224, 224)
    out = module4(tmp)
    print('model4:', out.shape)

    net = model()
    tmp = torch.randn(2, 3, 224, 224)
    out = net(tmp)
    print('net:', out.shape)

    # model=model18(5)
    # tmp=torch.randn(2,3,224,224)
    # out=model(tmp)
    # print('model:',out.shape)



if __name__=='__main__':
    main()
