#### Dataset
##### Tập dữ liệu VOC2007
###### Chạy đoạn code này tạo folder chứa dataset
```python
import os
import urllib.request
import zipfile
import tarfile

data_dir = "./dataset"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
target_path = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar")

if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)

    tar = tarfile.TarFile(target_path)
    tar.extractall(data_dir)
    tar.close
```

#### Training
##### Architecture mobilnetv1-ssd
```python
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
```

#### Inference
```python
!python Inference/inference_pipeline.ipynb
```



#### Retrain
##### Thêm đoạn code này vào trong file traning.py
###### Tải weights ở chỗ này [URL](https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth)

```python
#Khởi tạo model
net = create_mobilenetv1_ssd(num_classes = 21)
#pre-trained
net.init_from_pretrained_ssd('/content/ai4theblind/object_detection/Mobilenetv1-SSD/mobilenet-v1-ssd-mp-0_675.pth')
```
