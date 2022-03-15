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

coco_classes = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"]

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

data_path =  'data/coco/'
save_folder = 'trained_models'
log_path = 'tensorboard/SSD'

model = 'ssd'
num_epochs = 1000
batch_size = 16
multistep = [43,54]
lr = 0.0001
momentum = 0.9
weight_decay = 0.0005
nms_threshold = 0.5
num_workers = 6

#Load model
dboxes = generate_dboxes(model="ssd")
model = SSD(backbone=ResNet(), num_classes=len(coco_classes))

#Load dataset
train_set = CocoDataset(data_path, 2017, "train", SSDTransformer(dboxes, (300, 300), val=False))
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=num_workers)
test_set = CocoDataset(data_path, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)

encoder = encoder(dboxes)

lr = lr  * (batch_size / 32)
criterion = Loss(dboxes)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay,
                                nesterov=True)
scheduler = MultiStepLR(optimizer=optimizer, milestones=multistep, gamma=0.1)


#Lưu các giá trị: loss - acc - tạo folder log_path
if os.path.isdir(log_path):
    shutil.rmtree(log_path)
os.makedirs(log_path)

#Tạo folder save_folder
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
#Load weight ra
checkpoint_path = os.path.join(save_folder, "SSD.pth")

writer = SummaryWriter(log_path)

#Nếu có file weight thì load ra còn không epoch_first = 0
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    first_epoch = checkpoint["epoch"] + 1
    model.module.load_state_dict(checkpoint["model_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    optimizer.load_state_dict(checkpoint["optimizer"])
else:
    first_epoch = 0

for epoch in range(first_epoch, num_epochs):
    train(model, train_loader, epoch, writer, criterion, optimizer, scheduler)
    evaluate(model, test_loader, epoch, writer, encoder, nms_threshold)

    checkpoint = {"epoch": epoch,
                  "model_state_dict": model.module.state_dict(),
                  "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()}
    torch.save(checkpoint, checkpoint_path)