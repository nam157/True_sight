import sys
sys.path.append('C:/Users/nguye/OneDrive/Desktop/ai4theblind/SSD_ver2')
from data_loader.make_datapath import make_datapath_list
from data_loader.dataset import MyDataset, my_collate_fn
from data_loader.transform import DataTransform
from data_loader.extract_inform_annotation import Anno_xml
from total_network.model_mobilenetv2_ssd import SSDLite,MobileNetV2
from total_network.model_resnet50_ssd import SSD
from total_network.vgg_model import SSD_VGG
from utils.multiboxloss import MultiBoxLoss
import utils.cfg as cfg
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd

name_net = 'resnet_ssd'
root_path = "C:/Users/nguye/OneDrive/Desktop/ai4theblind/dataset/VOC2007"
weight_retrain = 'C:/Users/nguye/OneDrive/Desktop/ai4theblind/Total_SSD/weights/SSD.pth'


if name_net == 'vgg_ssd':
    net = SSD_VGG(phase='train',cfg = cfg.cfgs)
elif name_net == 'resnet_ssd':
    net = SSD(num_classes=21,phase = 'train')
elif name_net == 'mobi_ssd':
    net = SSDLite()




train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(root_path)
train_dataset = MyDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(cfg.input_size, cfg.color_mean), anno_xml=Anno_xml(cfg.classes))
val_dataset = MyDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(cfg.input_size, cfg.color_mean), anno_xml=Anno_xml(cfg.classes))

batch_size = 16
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)
dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

# images,target = next(iter(val_dataloader))
# print(images.shape)  #batch_size,channel,300,300
# print(target[0].shape) #xmin,ymin,xmax,ymax,label_id

DEVICE = torch.device('cpu')


net.load_state_dict(torch.load(weight_retrain),strict=False)

# MultiBoxLoss
criterion = MultiBoxLoss(jaccard_threshold=0.5, neg_pos=3, device=DEVICE)

# optimizer
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    # move network to GPU
    net.to(DEVICE)

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []
    for epoch in range(num_epochs+1):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        print("---"*20)
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("---"*20)
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                print("(Training)")
            else:
                #10 epoch minh se kiem dinh 1 lan
                if (epoch+1) % 10 == 0:
                    net.eval() 
                    print("---"*10)
                    print("(Validation)")
                else:
                    continue
            for images, targets in dataloader_dict[phase]:
                # move to GPU
                images = images.to(DEVICE)
                targets = [ann.to(DEVICE) for ann in targets]
                # init optimizer
                optimizer.zero_grad()
                #forward: dua anh vao trong mang cua minh
                with torch.set_grad_enabled(phase=="train"):
                    outputs = net(images)
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == "train":
                        loss.backward() # calculate gradient
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                        optimizer.step() # update parameters

                        if (iteration % 10) == 0:
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start
                            print("Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec".format(iteration, loss.item(), duration))
                            t_iter_start = time.time()
                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()
        t_epoch_end = time.time()
        print("---"*20)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}".format(epoch+1, epoch_train_loss, epoch_val_loss))           
        print("Duration: {:.4f} sec".format(t_epoch_end - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {"epoch": epoch+1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("./data/ssd_logs.csv")
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(), "./data/weights/ssd300_" + str(epoch+1) + ".pth")

num_epochs = 100
train_model(net, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)
