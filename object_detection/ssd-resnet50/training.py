import os
import shutil
import torch
from pycocotools.cocoeval import COCOeval
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from models import SSD, ResNet
from transform import SSDTransformer
from Loss import Loss
import numpy as np
from dataset import collate_fn, CocoDataset
from Default_boxes import generate_dboxes
import encoder
from torch.utils.tensorboard import SummaryWriter
import config
import yaml
import time





with open('Resnet-ssd/trainer_config.yaml', 'r', encoding="utf8") as stream:
    opt = yaml.safe_load(stream)
    #Load model
    dboxes = generate_dboxes(model="ssd")
    model = SSD(backbone=ResNet(), num_classes=len(config.coco_classes))

    #Load dataset
    train_set = CocoDataset(opt.data_path, 2017, "train", SSDTransformer(dboxes, (300, 300), val=False))
    train_loader = DataLoader(train_set,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    test_set = CocoDataset(opt.data_path, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))
    test_loader = DataLoader(test_set,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)

    dataloader_dict ={
                'train':train_loader,
                'val':test_loader
                    }
    DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    encoder = encoder(dboxes)

    lr = opt.lr  * (opt.batch_size / 32)
    criterion = Loss(dboxes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay,
                                    nesterov=True)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=opt.multistep, gamma=0.1)

    model.to(DEVICE)
    criterion.to(DEVICE)

    #Lưu các giá trị: loss - acc - tạo folder log_path
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    #Tạo folder save_folder
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    #Load weight ra
    checkpoint_path = os.path.join(opt.save_folder, "SSD.pth")

    writer = SummaryWriter(opt.log_path)

    #Nếu có file weight thì load ra còn không epoch_first = 0
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        first_epoch = checkpoint["epoch"] + 1
        model.module.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        first_epoch = 0
    

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []
    for epoch in range(first_epoch, opt.num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        print("---"*20)
        print("Epoch {}/{}".format(epoch, opt.num_epochs))
        print("---"*20)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                print("(Training)")
            else:
                #10 epoch minh se kiem dinh 1 lan
                if (epoch+1) % 10 == 0:
                    model.eval() 
                    print("---"*10)
                    print("(Validation)")
                else:
                    continue
            for images, _,_,gloc, glabel in dataloader_dict[phase]:
                # move to GPU
                images = images.to(DEVICE)
                gloc = gloc.to(DEVICE)
                glabel = glabel.to(DEVICE)
                # init optimizer
                optimizer.zero_grad()
                #forward: dua anh vao trong mang cua minh
                with torch.set_grad_enabled(phase=="train"):
                    ploc, plabel = model(images)
                    ploc, plabel = ploc.float(), plabel.float()
                    gloc = gloc.transpose(1, 2).contiguous()
                    loss = criterion(ploc, plabel, gloc, glabel)
    
                    if phase == "train":
                        loss.backward() # calculate gradient
                        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
                        optimizer.step() # update parameters

                        if (iteration % 10) == 0:
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start
                            print("Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec".format(iteration, loss.item(), duration))
                            t_iter_start = time.time()
                        epoch_train_loss += loss.item()
                        iteration += 1
                        writer.add_scalar("Train/Loss", loss.item(), epoch)
                    else:
                        epoch_val_loss += loss.item()
                        writer.add_scalar("val/Loss", loss.item(), epoch)
                        
        t_epoch_end = time.time()
        print("---"*20)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}".format(epoch+1, epoch_train_loss, epoch_val_loss))           
        print("Duration: {:.4f} sec".format(t_epoch_end - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {"epoch": epoch+1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
        logs.append(log_epoch)
        import pandas as pd
        df = pd.DataFrame(logs)
        df.to_csv("./data/ssd_logs.csv")
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        if ((epoch+1) % 10 == 0):
            checkpoint = {"epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()}
            torch.save(checkpoint, checkpoint_path)
        
