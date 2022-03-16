## Dataset

##### Tạo tập dữ liệu dataset coco 2014

```python
!git clone https://github.com/nam157/ai4theblind.git
!wget http://images.cocodataset.org/zips/train2014.zip
!wget http://images.cocodataset.org/zips/val2014.zip
!wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
!unzip "/content/train2014.zip" -d "/content/drive/MyDrive/ai4theblind/dataset"
!unzip "/content/val2014.zip" -d "/content/drive/MyDrive/ai4theblind/dataset/"
!unzip "/content/annotations_trainval2014.zip" -d "/content/drive/MyDrive/ai4theblind/dataset/"
```

##### Data loader
Dựa vào apis của tập dữ liệu coco, chúng ta trích xuất thông tin file annatations và images.
Để dữ liệu nó đa dạng hơn chúng cần transform data, và file transform được lấy từ https://github.com/amdegroot/ssd.pytorch/tree/master/utils
```python
%run dataset.py
```

## Training
##### Trong bài toán này, base network sử dụng mạng resnet 50 dùng trích xuất đặc trưng và kết hợp với mạng một số lớp.
* Chỉnh sửa tham số training cho phù hợp ở file trainer_config.yaml
```python
%run training.py
```
## Inference
```python
%run inference_pipeline.py
```
## Save weights
|Models                           | Folder                        |
|---------------------------------|-------------------------------|
|Resnet50-ssd                     |[Here](https://drive.google.com/drive/u/0/folders/1L6IrWa8QI6WBo15iM9Yl30-w0UeqCAql)|
