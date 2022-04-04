import sys
sys.path.insert(0,'../../object_detection')
from library import *
from test import Model
import os
import cv2,json


model = Model()


def focal_length_finder(measured_distance, real_width, width_in_rf):       
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

def distance_finder(focal_length, real_object_width, width_in_frmae):        
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


f = open('parameter.json', "r")
info_calculate = json.load(f)
info_calculate


info_focal_length = {}
for image_name in os.listdir('../distance_ver3/data_refers/'):
    img_path = os.path.join('../distance_ver3/data_refers/',image_name)
    ref_image = cv2.imread(img_path)
    frame,center_point_iphone,width,category = model.draw_object_info(ref_image)
    focal_length = focal_length_finder(info_calculate[category]['KNOWN_DISTANCE'],info_calculate[category]['KNOWN_WIDTH'],width)
    info_focal_length[category] = focal_length

print(info_focal_length)

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    loc,label,prob = model.pre_pare(frame)
    if len(loc) > 0:
        height,width,channel = frame.shape
        loc[:, 0::2] *= width
        loc[:, 1::2] *= height
        loc = loc.astype(np.int32)

        for box,lb,pb in zip(loc,label,prob):
            category = config.coco_classes[lb]
            color = config.colors[lb]
            xmin, ymin, xmax, ymax = box
            center_point = ((xmin + xmax) / 2, (ymin + ymax ) / 2)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            text_size = cv2.getTextSize(category + " : %.2f" % pb, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(frame, (xmin, ymin), (xmin + text_size[0] + 200, ymin + text_size[1] + 4), color,-1)
            cv2.putText(frame, category + " : %.2f" % pb,(xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,(255, 255, 255), 1)
            try:
                distance =  distance_finder(info_focal_length[category], info_calculate[category]['KNOWN_WIDTH'], xmax-xmin)
                cv2.circle(frame,(int(center_point[0]),int(center_point[1])),3,(255,255,0),2)
                cv2.putText(frame, f'Distance: {round(distance*2.54,2)}',(int(center_point[0]),int(center_point[1])), cv2.FONT_HERSHEY_PLAIN, 1,(123, 0, 255), 1)
                print(f'Loại:{category} có distance: {distance*2.54}' )
            except:
                print('ERROR')
        
    cv2.imshow('VIDEO',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite('refers_img_'+ category +'.jpg',frame)
        print("images saved!")


cv2.destroyAllWindows()


