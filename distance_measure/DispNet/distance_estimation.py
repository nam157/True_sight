
import sys

sys.path.append('C:/Users/nguye/OneDrive/Desktop/ai4theblind/Resnet-ssd/')
import numpy as np
import torch
from transform import SSDTransformer
import cv2
from PIL import Image
from Default_boxes import generate_dboxes
from encoder import Encoder
from models import SSD, ResNet
import time
import config
import os,pickle
#-------------------------------------------------------------------
model = SSD(backbone=ResNet())
checkpoint = torch.load(config.pretrained_model)
model.load_state_dict(checkpoint["model_state_dict"])
if torch.cuda.is_available():
    model.cuda()
model.eval()
dboxes = generate_dboxes()
transformer = SSDTransformer(dboxes, (300, 300), val=True)
encoder = Encoder(dboxes)
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
def pre_pare(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame, _, _, _ = transformer(frame, None, torch.zeros(1, 4), torch.zeros(1))
    frame = frame.to(device)
    with torch.no_grad():
        ploc, plabel = model(frame.unsqueeze(dim=0))
        result = encoder.decode_batch(ploc, plabel, config.nms_threshold, 20)[0]
        loc, label, prob = [r.cpu().numpy() for r in result]
        best = np.argwhere(prob > config.cls_threshold).squeeze(axis=1)
        loc = loc[best]
        label = label[best]
        prob = prob[best]
    
        return loc,label,prob
#----------------------------------------------------------------------
def depth_to_distance(depth):
        # loaded_model = pickle.load(open('model_distance.sav', 'rb'))
        # depth = np.array([[depth]])
        # return loaded_model.predict(depth)
    return ((-39.89694666*(depth**2)) + (-53.87434965*depth) + 111.01373772611525)
    
#-----------------------------------------------------------------------
# Load a MiDas model for depth estimation
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

#------------------------------------------------------------------------
cap = cv2.VideoCapture(1)

while True:
    ret,img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    loc,label,prob = pre_pare(img)
    if len(loc) > 0:
        height, width, _ = img.shape
        loc[:, 0::2] *= width
        loc[:, 1::2] *= height
        loc = loc.astype(np.int32)
        for box, lb, pr in zip(loc, label, prob):
            category = config.coco_classes[lb]
            color = config.colors[lb]
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            center_point = ((xmin + xmax) / 2, (ymin + ymax ) / 2)
            text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color,-1)
            cv2.putText(img, category + " : %.2f" % pr,(xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,(255, 255, 255), 1)

    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),size=img.shape[:2],   mode="bicubic",align_corners=False,).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    try:
        depth = depth_map[int(center_point[1]),int(center_point[0])]
        print(depth)
        distance = depth_to_distance(depth)
        cv2.circle(img,(int(center_point[0]),int(center_point[1])),3,(255,255,0),2)
        cv2.putText(img, f'Distance: {round(distance,2)}',(int(center_point[0]),int(center_point[1])), cv2.FONT_HERSHEY_PLAIN, 1,(123, 0, 255), 1)
        print('Convert depth map to distance: ',distance)
    except:
        print('Không tìm thấy bbox trong khung hình')

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)
    
    cv2.imshow('test',img)
    cv2.imshow('depth_map',depth_map)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

