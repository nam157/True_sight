### Prepare
#### dataset VOC 2012
```python
exec(open("ai4theblind/object_detection/data/data_loader/prepare_data.py").read())
```
### weights VGG16
```python
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

### Change path in file train.py
```python
root_path = "..."
```

### Run training 
```
python train.py
```

