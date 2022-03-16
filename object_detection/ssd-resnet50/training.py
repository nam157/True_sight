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





def train(model, train_loader, epoch, writer, criterion, optimizer, scheduler):
    model.train()
    num_iter_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader)
    scheduler.step()
    for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
        if torch.cuda.is_available():
            img = img.cuda()
            gloc = gloc.cuda()
            glabel = glabel.cuda()

        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()
        loss = criterion(ploc, plabel, gloc, glabel)

        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))

        writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def evaluate(model, test_loader, epoch, writer, encoder, nms_threshold):
    model.eval()
    detections = []
    category_ids = test_loader.dataset.coco.getCatIds()
    for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):
        print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            # Get predictions
            ploc, plabel = model(img)
            ploc, plabel = ploc.float(), plabel.float()

            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue

                height, width = img_size[idx]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    detections.append([img_id[idx], loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,
                                       (loc_[3] - loc_[1]) * height, prob_,
                                       category_ids[label_ - 1]])
    detections = np.array(detections, dtype=np.float32)

    coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)


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

    encoder = encoder(dboxes)

    lr = opt.lr  * (opt.batch_size / 32)
    criterion = Loss(dboxes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay,
                                    nesterov=True)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=opt.multistep, gamma=0.1)


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

    for epoch in range(first_epoch, opt.num_epochs):
        train(model, train_loader, epoch, writer, criterion, optimizer, scheduler)
        evaluate(model, test_loader, epoch, writer, encoder, opt.nms_threshold)

        checkpoint = {"epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)