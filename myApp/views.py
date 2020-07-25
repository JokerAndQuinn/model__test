import json
from django.http import HttpResponse
from model import model
import torch
from torchvision import transforms
from PIL import  Image
from torch.utils.data import DataLoader
from django.http import QueryDict
# Create your views here.
from django.views.decorators.csrf import csrf_exempt




tf = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

net = model()
net.load_state_dict(torch.load('best.mdl',map_location='cpu'))

@csrf_exempt
def my_api(request):
    dic = {}
    if request.method == 'POST':
        data = request.FILES.get('image')
        # img=data.read()
        img = tf(data)

        img=img.unsqueeze(0)
        # pred=img.shape
        logits = net(img)
        pred = logits.argmax()
        dic['infer'] = pred.item()
        return HttpResponse(json.dumps(dic))
    else:
        dic['message'] = '方法错误'
        return HttpResponse(json.dumps(dic, ensure_ascii=False))


@csrf_exempt
def my_api1(request):
    dic = {}
    if request.method == 'POST':
        data = request.FILES.get('image')
        name0 = data.name
        # img=data.read()

        with open('train.json', 'r') as t:
            t = json.load(t)
        t=t['annotations']
        for i in range(len(t)):
            labelname=t[i]['name'].split('/')[-1]
            if name0==labelname:
                label=t[i]['annotation']
                break
            else:
                label=0
                    # self.name2label[name] = labelset[i]['num']
        img = tf(data)
        img=img.unsqueeze(0)
        # pred=img.shape
        logits = net(img)
        pred = logits.argmax()
        dic['infer'] = pred.item()
        dic['label']=label
        return HttpResponse(json.dumps(dic))
    else:
        dic['message'] = '方法错误'
        return HttpResponse(json.dumps(dic, ensure_ascii=False))