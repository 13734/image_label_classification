from dataset import Dataset_Loader
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json


with open("project.json","r",encoding='utf-8') as d:
    str_json = d.read()
    train_json = json.loads(str_json)

num_classes = train_json["num_labels"]
# %matplotlib inline
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
treshold =0.35
use_best =True
def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 0)
    return softmax_x



with open('tags_all.txt', 'r', encoding='utf-8') as f:
    tags = f.readlines()
    tags = [ti.strip() for ti in tags]


if __name__ == "__main__":
    sigmoid = nn.Sigmoid()
    test_list = 'val.txt'
    test_data = Dataset_Loader(test_list, train_flag=False)
    test_loader = DataLoader(dataset=test_data, num_workers=1, pin_memory=True, batch_size=1)
    model = models.resnet18(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, num_classes)
    model = model.cuda()
    # 加载训练好的模型
    checkpoint = torch.load('model_best_checkpoint_resnet18.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for i, (image, label) in enumerate(test_loader):
        src = image.numpy()
        src = src.reshape(3, 224, 224)
        src = np.transpose(src, (1, 2, 0))
        image = image.cuda()
        label = label.cuda()
        pred = model(image)
        #pred = pred.data.cpu().numpy()[0]
        pred = pred.data.cpu()[0]
        pred = sigmoid(pred)
        

        res_dict =dict()
        res_dict2 = dict()

        print(tags[0])
        for i,tag in enumerate(tags):
            res_dict[tag] = pred[i]
        for tag in tags:
            if res_dict[tag] > treshold:
                res_dict2[tag] = res_dict[tag]

        plt.imshow(src)
        print('预测结果：', res_dict2)
        plt.show()

