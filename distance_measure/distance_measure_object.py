#Object detection
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
import pyttsx3
import time
import config

#Measure distance
import measure as tri
import calibration

#--------------- Load model SSD------------------
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
#---------------Text to speech------------------
engine = pyttsx3.init()
#---------------Distance measure-----------------

# Gọi 2 camera ra 
cap_right = cv2.VideoCapture(0)                    
cap_left =  cv2.VideoCapture(1)


# Stereo vision các thông số cần setup
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B = 9               #Distance between the cameras [cm]
f = 8               #Camera lense's focal length [mm]
alpha = 56.6        #Camera field of view in the horisontal plane [degrees]


while(cap_right.isOpened() and cap_left.isOpened()):
    #Đọc 2 camera ra
    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)
    #Kiem tra nó hoạt động không, nếu không thì dừng còn có thì object detection
    if not succes_right or not succes_left:                    
        break
    else:
        start = time.time()
        #Copy 2 ảnh ban đầu chưa transform
        output_left_frame = frame_left.copy()
        output_right_frame = frame_right.copy()
        #Bắt đầu sử dụng thuật toán ssd: xử lý ảnh đầu vào và đưa ra ảnh predict
        loc_left,label_left,prob_left = pre_pare(frame_left)
        loc_right,label_right,prob_right = pre_pare(frame_right)

        #Lưu tâm của bbox của 2 camera
        centers = {}
        #Xử lý bên camera trái
        if len(loc_left) > 0:
            height, width, channel = output_left_frame.shape
            #transform bbox ban đầu bằng cách nhân với w,h ban đầu
            loc_left[:, 0::2] *= width
            loc_left[:, 1::2] *= height
            loc_left = loc_left.astype(np.int32)
            #Load bbox, label, prob
            for box, lb, pr in zip(loc_left, label_left, prob_left):
                category = config.coco_classes[lb]
                color = config.colors[lb]
                xmin, ymin, xmax, ymax = box
                #Tạo tâm bbox
                center_point_left = (xmin + xmax / 2, ymin + ymax / 2)
                #Vẽ bbox cho object và vẽ classes tương ứng
                cv2.rectangle(output_left_frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(output_left_frame, category + " : %.2f" % pr,(xmin, ymin+4), cv2.FONT_HERSHEY_PLAIN, 1,(255, 255, 255), 1)
                centers['center_point_left'] = center_point_left
        #Xử lý bên camera phái
        if len(loc_right) > 0:
            height, width, channel = output_right_frame.shape
            loc_right[:, 0::2] *= width
            loc_right[:, 1::2] *= height
            loc_right = loc_right.astype(np.int32)
            for box, lb, pr in zip(loc_right, label_left, prob_left):
                category = config.coco_classes[lb]
                color = config.colors[lb]
                xmin, ymin, xmax, ymax = box
                center_point_right = (xmin + xmax / 2, ymin + ymax / 2)
                cv2.rectangle(output_right_frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(output_right_frame, category + " : %.2f" % pr,(xmin, ymin+4), cv2.FONT_HERSHEY_PLAIN, 1,(255, 255, 255), 1)
                centers['center_point_right'] = center_point_right
        print(centers)
        #Nếu nhưng mà centers rỗng tức là ko có bbox nào thì in ra lỗi
        if len(centers) == 0:
            cv2.putText(output_right_frame, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            cv2.putText(output_left_frame, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        else:
            #Tính toán khoảng cách dự trên công thức đã có: z = (baseline*f_pixel)/disparity (cm)
            depth = tri.find_depth(centers['center_point_right'], centers['center_point_left'], output_right_frame, output_left_frame, B, f, alpha)
            cv2.putText(output_right_frame, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            cv2.putText(output_left_frame, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)


        cv2.imshow("frame right", output_right_frame) 
        cv2.imshow("frame left", output_left_frame)


        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()