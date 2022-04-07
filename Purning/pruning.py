import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.25),
            #2
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.25),
            #4
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #5
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.25),
            #6
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.25),
            #7
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #8
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.25),
            #9
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.25),
            #10
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #11
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.25),
            #12
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.25),
            #13
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.AvgPool2d(kernel_size=1,stride=1),
            )
        self.classifier = nn.Sequential(
            #14
            nn.Linear(512,4096),
            nn.ReLU(True),
            nn.Dropout(0.25),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.25),conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
            )
        #self.classifier = nn.Linear(512, 10)
 
    def forward(self, x):
        out = self.features(x)
#        print(out.shape)
        out = out.view(out.size(0), -1)
#        print(out.shape)
        out = self.classifier(out)
#        print(out.shape)
        return out
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = nameconv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        #print("pred: ",pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            #print("target",target)
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        images, target = images.to(device), target.to(device)
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.5f ms' % (elapsed/num_images*1000))
    return elapsed

def load_model(model_file):
    model = VGG16()
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
""""""
def prepare_data_loaders():

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='/home/xdd44/Desktop/Codfish/quantization/data', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='/home/xdd44/Desktop/Codfish/quantization//data', train=False, download=False, transform=transform_test)

    train_sampler = torch.utils.data.RandomSampler(trainset)
    test_sampler = torch.utils.data.SequentialSampler(testset)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=eval_batch_size, sampler=test_sampler)

    return trainloader, testloader

saved_model_dir = '/home/xdd44/Desktop/Codfish/pruning/'
float_model_file = 'test0124_module.pth'
scripted_float_model_file = 'VGG16_pruning_scripted.pth'
scripted_default_pruned_model_file = 'VGG16_pruning_scripted.pth'

train_batch_size = 128
eval_batch_size = 100

data_loader, data_loader_test = prepare_data_loaders()
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to(device)

#print("origin model ",float_model)
float_model.eval()
num_eval_batches = 200

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg,top5.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

model = load_model(saved_model_dir + float_model_file).to(device)
parameters_to_prune = (
    (model.features[0], 'weight'),
    (model.features[4], 'weight'),
    (model.features[8], 'weight'),
    (model.features[12], 'weight'),
    (model.features[16], 'weight'),
    (model.features[20], 'weight'),
    (model.features[24], 'weight'),
    (model.features[28], 'weight'),
    (model.features[32], 'weight'),
    (model.features[36], 'weight'),
    (model.features[40], 'weight'),
    (model.features[44], 'weight'),
    (model.features[48], 'weight'),
    (model.classifier[0], 'weight'),
    (model.classifier[3], 'weight'),
    (model.classifier[6], 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.8,
)
for module in model.modules():
    if isinstance(module,torch.nn.Conv2d):
        prune.remove(module,'weight')
    if isinstance(module,torch.nn.Linear):
        prune.remove(module,'weight')
print("Size of pruned model")
print_size_of_model(model)

top1, top5 = evaluate(model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg,top5.avg))
torch.jit.save(torch.jit.script(model), saved_model_dir + scripted_default_pruned_model_file)

run_benchmark(saved_model_dir + scripted_float_model_file,data_loader_test)
run_benchmark(saved_model_dir + scripted_default_pruned_model_file,data_loader_test)