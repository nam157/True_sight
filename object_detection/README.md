



### Prepare
#### dataset VOC 2012
```python
exec(open("./data/data_loader/prepare_data.py").read())
```
#### weights VGG16
```python
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

### Change path in file train.py
```python
root_path = "..."
```

### Run training 
```python
exec(open("./train.py").read())
```
### Run test
```python
exec(open("Inference/inference.py").read())
```
### Notebook-Kaggle 
* [notebook1](https://www.kaggle.com/acousticmusic/ai4theblind/notebook)

### Result
|Input                                  |Output                                       |
|-------------------------------------- |---------------------------------------------|
|![004545](https://user-images.githubusercontent.com/72034584/155994827-c36db51b-b368-4628-9a61-b4e80db4b005.jpg)|![img](https://user-images.githubusercontent.com/72034584/156599512-3087476a-11a9-43ad-9ad4-d686f4f90991.jpg)|
|![000067](https://user-images.githubusercontent.com/72034584/155995173-f9a3c173-6238-46a5-885e-2288cff42cce.jpg)|![img_0](https://user-images.githubusercontent.com/72034584/155995195-153d694b-0650-4b8d-bf51-9c82d5d634f7.jpg)|
|![images](https://user-images.githubusercontent.com/72034584/155995323-b01695a7-dcc5-4ae0-a110-05f8a8f84712.jpg)|![img_0](https://user-images.githubusercontent.com/72034584/155995355-d20f2c33-4dfb-41d3-9904-112eec3e37d5.jpg)|

https://user-images.githubusercontent.com/72034584/155999858-39b79ab9-116c-40d6-b358-58a4953d183a.mp4


