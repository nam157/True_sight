import sys
sys.path.append('C:/Users/nguye/OneDrive/Desktop/ai4theblind/Mobilenetv1-SSD/')
from models.mobilenet_model import create_mobilenetv1_ssd
import cv2
from Inference.predictor import Predictor,create_mobilenetv1_ssd_predictor



class_names = [name.strip() for name in open('C:/Users/nguye/OneDrive/Desktop/ai4theblind/Mobilenetv1-SSD/voc-model-labels.txt').readlines()]
num_classes = len(class_names)

#Load model-weight
net = create_mobilenetv1_ssd(num_classes=21,is_test = True)
model_path = "C:/Users/nguye/OneDrive/Desktop/ai4theblind/Mobilenetv1-SSD/save_weights/mobilenet-v1-ssd-mp-0_675.pth"
net.load(model_path)
#Load predictor
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)


image_path = "C:/Users/nguye/OneDrive/Desktop/ai4theblind/SSD_ver2/test_images/000187.jpg"
orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.7)


for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,(int(box[0]) + 20, int(box[1]) + 40),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),2)
cv2.imshow('test',orig_image)
cv2.waitKey(0)
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, orig_image)