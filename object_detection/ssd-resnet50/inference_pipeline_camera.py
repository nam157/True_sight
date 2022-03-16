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

import pyttsx3
import time
import config





model = SSD(backbone=ResNet())
checkpoint = torch.load(config.pretrained_model)
model.load_state_dict(checkpoint["model_state_dict"])
if torch.cuda.is_available():
        model.cuda()

model.eval()
dboxes = generate_dboxes()
transformer = SSDTransformer(dboxes, (300, 300), val=True)
cap = cv2.VideoCapture(0)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
encoder = Encoder(dboxes)




## Text to speech
# engine = pyttsx3.init()



while True:
    ref,frame = cap.read()    
    output_frame = np.copy(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame, _, _, _ = transformer(frame, None, torch.zeros(1, 4), torch.zeros(1))

    if torch.cuda.is_available():
        frame = frame.cuda()
    with torch.no_grad():
        ploc, plabel = model(frame.unsqueeze(dim=0))
        result = encoder.decode_batch(ploc, plabel, config.nms_threshold, 20)[0]
        loc, label, prob = [r.cpu().numpy() for r in result]
        best = np.argwhere(prob > config.cls_threshold).squeeze(axis=1)
        loc = loc[best]
        label = label[best]
        prob = prob[best]
        if len(loc) > 0:
            loc[:, 0::2] *= width
            loc[:, 1::2] *= height
            loc = loc.astype(np.int32)
            for box, lb, pr in zip(loc, label, prob):
                category = config.coco_classes[lb]
                color = config.colors[lb]
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_frame, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color,-1)
                cv2.putText(output_frame, category + " : %.2f" % pr,(xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,(255, 255, 255), 1)

                # engine.say(category)
                # engine.runAndWait()           

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(output_frame,'FPS:' + str(round(fps,3)),(0,20), cv2.FONT_HERSHEY_PLAIN, 1,(255, 0, 255), 1)
    cv2.imshow('image',output_frame)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
