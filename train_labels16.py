#from dataset import Garbage_Loader
from dataset import Dataset_Loader
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models
import  torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.optim as optim
import torch
import time
import os
import shutil
from torch.cuda.amp import autocast 
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"





from tensorboardX import SummaryWriter

def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]
def calculate_acuracy_mode_one(model_pred, labels,rate):
    sigm = nn.Sigmoid()
    model_pred = sigm(model_pred)
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = rate
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0, 0
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    # 模型预测的结果中有多少个是正确的
    precision = true_predict_num / pred_one_num
    # 模型预测正确的结果中，占所有真实标签的数量
    recall = true_predict_num / target_one_num

    return precision.item(), recall.item()

# 计算准确率——方式2
# 取预测概率最大的前top个标签，作为模型的预测结果
def calculate_acuracy_mode_two(model_pred, labels):
    sigm = nn.Sigmoid()
    model_pred = sigm(model_pred)
    # 取前top个预测结果作为模型的预测结果
    precision = 0
    recall = 0
    top = 5
    # 对预测结果进行按概率值进行降序排列，取概率最大的top个结果作为模型的预测结果
    pred_label_locate = torch.argsort(model_pred, descending=True)[:, 0:top]
    for i in range(model_pred.shape[0]):
        temp_label = torch.zeros(1, model_pred.shape[1])
        temp_label[0,pred_label_locate[i]] = 1
        target_one_num = torch.sum(labels[i])
        true_predict_num = torch.sum(temp_label * labels[i])
        # 对每一幅图像进行预测准确率的计算
        precision += true_predict_num / top
        # 对每一幅图像进行预测查全率的计算
        recall += true_predict_num / target_one_num
    return precision, recall
def accuracy2(output, target, topk=(1,)):
    """
        计算topk的准确率
    """
    with torch.no_grad():


        batch_size = target.size(0)

        num = int(target.sum()/batch_size)+1
        maxk = max(topk) *  num

        _, pred = output.topk(maxk, 1, True, True)

        class_to = 0

        res = []
        countl =[0] * len(topk)
        for single_x , single_y in zip(pred, target):
            for k_idx in range(len(topk)):
                k = topk[k_idx]
                for x_h in single_x[:num*(k_idx+1)-1]:
                    x_h = int(x_h)
                    if single_y[x_h] >0.9:
                        countl[k_idx] += 1
        for num_i in countl:
            res.append(num_i/target.sum())
        return res, class_to



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
        根据 is_best 存模型，一般保存 valid acc 最好的模型
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_' + filename)


def train(train_loader, model, criterion, optimizer, epoch, writer):
    """
        训练代码
        参数：
            train_loader - 训练集的 DataLoader
            model - 模型
            criterion - 损失函数
            optimizer - 优化器
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_recall = AverageMeter()
    top2_recall = AverageMeter()
    top1_prec = AverageMeter()
    top2_prec= AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    avg_meter = AverageMeter2()

    

    for i, (input, target) in enumerate(train_loader):
       
        input = input.cuda()
        target = target.cuda(non_blocking=True) #np.array?
        #forward
        
        with autocast():
            output = model(input)

            optimizer.zero_grad() #added

            loss = criterion(output, target)
        #loss = F.multilabel_soft_margin_loss(output,target)
        
        #back and update
        loss.backward()
        avg_meter.add({"loss":loss.item()})
        optimizer.step()

        
        # measure accuracy and record loss
        #[prec1, prec5], class_to = accuracy2(output, target, topk=(1, 2))

        prec1, recall1 = calculate_acuracy_mode_one(output, target,rate =0.5)
        prec2, recall2 = calculate_acuracy_mode_one(output, target, rate=0.35)

        losses.update(loss.item(), input.size(0))
        top1_recall.update(recall1, input.size(0))
        top2_recall.update(recall2, input.size(0))
        top1_prec.update(prec1, input.size(0))
        top2_prec.update(prec2, input.size(0))
        # measure elapsed time
        
        time_check5 = time.time()
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_recall.val:.3f} ({top1_recall.avg:.3f})\t'
                  'Prec@2 {top2_recall.val:.3f} ({top2_recall.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1_recall=top1_recall, top2_recall=top2_recall))
    scheduler.step()
    writer.add_scalar('loss/train_loss', losses.val, global_step=epoch)
   

def validate(val_loader, model, criterion, epoch, writer, phase="VAL"):
    """
        验证代码
        参数：
            val_loader - 验证集的 DataLoader
            model - 模型
            criterion - 损失函数
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_recall = AverageMeter()
    top2_recall = AverageMeter()
    top1_prec = AverageMeter()
    top2_prec= AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            with autocast():
                output = model(input)
                loss = criterion(output, target)
            #loss = F.multilabel_soft_margin_loss(output,target)

            # measure accuracy and record loss

            #[prec1, prec5], class_to = accuracy(output, target, topk=(1, 5))

            prec1, recall1 = calculate_acuracy_mode_one(output, target, rate=0.5)
            prec2, recall2 = calculate_acuracy_mode_one(output, target, rate=0.35)

            losses.update(loss.item(), input.size(0))
            top1_recall.update(recall1, input.size(0))
            top2_recall.update(recall2, input.size(0))
            top1_prec.update(prec1, input.size(0))
            top2_prec.update(prec2, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test-{0}: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1_recall.val:.3f} ({top1_recall.avg:.3f})\t'
                      'Prec@2 {top2_recall.val:.3f} ({top2_recall.avg:.3f})'.format(
                    phase, i, len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1_recall=top1_recall, top2_recall=top2_recall))

        print(' * {} Prec@1 {top1_recall.avg:.3f} Prec@5 {top2_recall.avg:.3f}'
                  .format(phase, top1_recall=top1_recall, top2_recall=top2_recall))

    writer.add_scalar('loss/valid_loss', losses.val, global_step=epoch)
    return top1_recall.avg, top2_recall.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeter2(object):
    def __init__(self,*keys):
        self.__data =dict()
        for k in keys:
            self.__data[k] =[0.0,0]
    def add(self,dict):
        for k,v in dict.items():
            if k not in self.__data:
                self.__data[k] =[0.0,0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self,*keys):
        if len(keys) ==1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            vlist =[self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(vlist)

    def pop(self,key = None):
        if key in None:
            for k in self.__data.keys():
                self.__data[k] = [0.0,0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0,0]
            return v




if __name__ == "__main__":
    # -------------------------------------------- step 0/4 : 加载记录 ---------------------------
    with open("project.json","r",encoding='utf-8') as d:
        str_json = d.read()
    train_json = json.loads(str_json)
    num_classes = train_json["num_labels"]
    USEBEST =train_json["use_best"]
    batch_size = train_json["batch_size"]
    epochs =train_json["epochs"] 
    # -------------------------------------------- step 1/4 : 加载数据 ---------------------------
    train_dir_list = 'train.txt'
    valid_dir_list = 'val.txt'
    #batch_size = 256
    #epochs =40 
    #num_classes = 1000
    train_data = Dataset_Loader(train_dir_list, train_flag=True,num_classes=num_classes)
    valid_data = Dataset_Loader(valid_dir_list, train_flag=False,num_classes=num_classes)
    train_loader = DataLoader(dataset=train_data, num_workers=8, pin_memory=True, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, num_workers=8, pin_memory=True, batch_size=batch_size)
    train_data_size = len(train_data)
    print('训练集数量：%d' % train_data_size)
    valid_data_size = len(valid_data)
    print('验证集数量：%d' % valid_data_size)
    # ------------------------------------ step 2/4 : 定义网络 ------------------------------------
    #model = models.resnet50(pretrained=True)

    model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
    fc_inputs = model.fc.in_features # 2048 |50 
    model.fc = nn.Linear(fc_inputs, num_classes)
    if USEBEST: #加载既有模型
        print("load from local")
        checkpoint = torch.load('model_best_checkpoint_resnet50.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()


    # ------------------------------------ step 3/4 : 定义损失函数和优化器等 -------------------------
    lr_init = 0.00022
    lr_stepsize = 20
    weight_decay = 0.001
    #criterion = F.multilabel_soft_margin_loss().cuda()
    criterion = F.multilabel_soft_margin_loss
    optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)

    writer = SummaryWriter('runs/resnet50')
    # ------------------------------------ step 4/4 : 训练 -----------------------------------------
    best_prec0 = 0
    for epoch in range(epochs):
        #scheduler.step()
        train(train_loader, model, criterion, optimizer, epoch, writer)
        # 在验证集上测试效果
        valid_prec1, valid_prec5 = validate(valid_loader, model, criterion, epoch, writer, phase="VAL")
        valid_prec0 =valid_prec1  + valid_prec5 
        is_best = valid_prec0 > best_prec0
        best_prec0 = max(valid_prec0, best_prec0)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec0,
            'optimizer': optimizer.state_dict(),
        }, is_best,
            filename='checkpoint_resnet50.pth.tar')
    writer.close()
