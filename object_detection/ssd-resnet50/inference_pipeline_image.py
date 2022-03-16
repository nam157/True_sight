import sys
sys.path.append('C:/Users/nguye/OneDrive/Desktop/ai4theblind/SSD-pytorch/src')
import numpy as np
import torch
from transform import SSDTransformer
import cv2
from PIL import Image

from Default_boxes import generate_dboxes
from encoder import Encoder
from models import SSD, ResNet

import config

model = SSD(backbone=ResNet())
checkpoint = torch.load(config.pretrained_model)
model.load_state_dict(checkpoint["model_state_dict"])
if torch.cuda.is_available():
        model.cuda()
model.eval()
dboxes = generate_dboxes()
transformer = SSDTransformer(dboxes, (300, 300), val=True)
image = Image.open('./data_test/000104.jpg').convert("RGB")
image, _, _, _ = transformer(image, None, torch.zeros(1, 4), torch.zeros(1))
encoder = Encoder(dboxes)
if torch.cuda.is_available():
    image = image.cuda()
with torch.no_grad():
    p_loc, p_label = model(image.unsqueeze(dim=0))
    result = encoder.decode_batch(p_loc, p_label, config.nms_threshold, 20)[0]
    loc, label, prob = [r.cpu().numpy() for r in result]
    best = np.argwhere(prob > config.cls_threshold).squeeze(axis=1)
    loc = loc[best]
    label = label[best]
    prob = prob[best]
    img = cv2.imread('./data_test/000133.jpg')
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
            text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color,-1)
            cv2.putText(img, category + " : %.2f" % pr,(xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,(255, 255, 255), 1)

    cv2.imshow('predict_image',img)
    cv2.waitKey(0)
