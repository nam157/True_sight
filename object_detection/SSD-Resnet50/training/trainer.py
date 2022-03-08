import os
import shutil
from argparse import ArgumentParser

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.model import SSD, SSDLite, ResNet, MobileNetV2
import utils.config as config
from utils.encode import Encoder
from utils.default_boxes import generate_dboxes
from data_loader.transform import SSDTransformer
from utils.loss import Loss
from train import train
from eval import evaluate
from data_loader.dataset import collate_fn, DatasetCOCO




dboxes = generate_dboxes(model='ssd')
model = SSD(backbone=ResNet(),num_classes=len(config.coco_classes))

trainset = DatasetCOCO(config.root_path,year = 2014, mode= 'train' , transform=SSDTransformer(dboxes,(300,300),val=False))
valset = DatasetCOCO(config.root_path,year = 2014, mode= 'val' , transform=SSDTransformer(dboxes,(300,300),val=True))


train_params =     {"batch_size": config.batch_size ,
                    "shuffle": True,
                    "drop_last": False,
                    "num_workers": config.num_workers,
                    "collate_fn": collate_fn}

test_params =     {"batch_size": config.batch_size ,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": config.num_workers,
                   "collate_fn": collate_fn}

# train_loader = DataLoader(trainset,batch_size=config.batch_size,shuffle=True,drop_last=False,num_workers=config.num_workers,collate_fn=collate_fn)
# test_loader = DataLoader(valset,batch_size=config.batch_size,shuffle=False,drop_last=False,num_workers=config.num_workers,collate_fn=collate_fn)

train_loader = DataLoader(trainset, **train_params)
test_loader = DataLoader(valset, **test_params)

encoder = Encoder(dboxes)

config.lr = config.lr  * (config.batch_size / 32)
criterion = Loss(dboxes)

optimizer = torch.optim.SGD(model.parameters(), lr = config.lr,momentum=config.momentum,weight_decay=config.weight_decay,nesterov=True)
scheduler = MultiStepLR(optimizer=optimizer,milestones=config.multistep,gamma = 0.1)

#Tao folder chua log trong qua trinh traning
if os.path.isdir(config.log_path):
        shutil.rmtree(config.log_path)
    os.makedirs(config.log_path)

#Thu muc luu weight
if not os.path.isdir(config.save_folder):
    os.makedirs(config.save_folder)
    checkpoint_path = os.path.join(config.save_folder, "SSD.pth")

writer = SummaryWriter(config.log_path)

#Kiem tra co weight pre-train
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    first_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    optimizer.load_state_dict(checkpoint["optimizer"])
else:
    first_epoch = 0

#Training
for epoch in range(first_epoch, config.epochs):
    train(model, train_loader, epoch, writer, criterion, optimizer, scheduler)
    evaluate(model, test_loader, epoch, writer, encoder, config.nms_threshold)

    checkpoint = {"epoch": epoch,
                      "model_state_dict": model.module.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
    torch.save(checkpoint, checkpoint_path)







