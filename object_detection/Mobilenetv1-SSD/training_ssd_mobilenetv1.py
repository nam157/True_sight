from sklearn import datasets
from data_loader.data_preprocessing import TrainAugmentation,TestTransform
from models.mobilenet_model import create_mobilenetv1_ssd
from Loss.multibox_loss import MultiboxLoss
import models.config_mobilenetv1 as config
import torch
from data_loader.make_data import VOCDataset
from torch.utils.data import DataLoader, ConcatDataset
from data_loader.prior import MatchPrior
import os

# dataloader
# network -> SSD300
# loss -> MultiBoxLoss
# optimizer
# training, validation

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            print(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == "__main__":
    net = create_mobilenetv1_ssd(num_classes = 21)

    #Config data
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    #Load dataset
    #1. Load data train
    root_path = 'C:/Users/nguye/OneDrive/Desktop/ai4theblind/dataset/VOC2007'
    datasets = VOCDataset(root_path,transform=train_transform,target_transform=target_transform)
    train_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(train_dataset, batch_size = 16,num_workers=6,shuffle=True)

    #2. Load data validation
    val_dataset = VOCDataset(root_path, transform=test_transform, target_transform=target_transform, is_test=True)
    val_loader = DataLoader(val_dataset, batch_size = 16,num_workers=6,shuffle=True)


    #Loss, optimizer
    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9,
                                weight_decay=5e-4)

    #Training
    num_epochs = 1000
    for epoch in range(1,num_epochs):
        train(train_loader, net, criterion, optimizer,device=DEVICE, debug_steps=100, epoch=epoch)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            print(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join('../Mobilenetv1-SSD/save_weights', f"Epoch-{epoch}.pth")
            net.save(model_path)
