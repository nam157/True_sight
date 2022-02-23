
from utils.augmentation import Compose,ConvertFromInts,ToAbsoluteCoords,\
     PhotometricDistort, Expand, RandomSampleCrop,RandomMirror, ToPercentCoords, Resize, SubtractMeans
from extract_info_annotation import *
from make_data_path import *
from library import *


class DataTransform:
    """
    *input_size: kich thuoc cua image
    *color_mean: (104, 117, 123)
    """
    def __init__(self,input_size,color_mean):
        self.data_transoform = {
            "train": Compose(
            [ConvertFromInts(), #Chuyen doi ve float32
            ToAbsoluteCoords(), #Tra ve thong so pixel ban dau
            PhotometricDistort(), #Change color by random
            Expand(color_mean),
            RandomSampleCrop(), #Cat phan bat ky trong anh
            RandomMirror(), #Xoay anh thanh nguoc lai
            ToPercentCoords(), #Chuan hoa ve 0 - 1
            Resize(input_size),
            SubtractMeans(color_mean),# Trừ đi mean của change BGR
            ]),
            "val" :Compose([ConvertFromInts(),Resize(input_size),SubtractMeans(color_mean)])
        }
    def __call__(self,img,phase,boxes,labels):
        return self.data_transoform[phase](img,boxes,labels)



#Test function transform
if __name__ == "__main__":

    root_path = './VOC2007/'
    train_image_list,train_annotation_list,val_image_list,val_annotation_list = make_data_list(root_path)
    img_file_path = train_image_list[1]

    img = cv2.imread(img_file_path) #BGR
    h,w,c = img.shape

    classes = ['aeroplane','bicycle','bird','boat',
            'bottle','bus','car','cat','chair','cow','diningtable','dog',
            'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    anno_xml = Annotation_xml(classes=classes)

    anno_info_list = anno_xml.__call__(train_annotation_list[1],w,h)
    print(anno_info_list)

    #plot ảnh
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # mặc định của matplotlib là RGB
    # plt.show()

    #prepare data transform
    color_mean = (104, 117, 123)
    input_size = 300
    transform = DataTransform(input_size,color_mean)

    #phase: train
    phase  = 'train'
    img_transformed, boxes, labels  = transform(img,phase,anno_info_list[:,:4],anno_info_list[:,4])
    print(img_transformed.shape)
    cv2.imshow("image test",cv2.cvtColor(img_transformed,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

    #phase: val
    phase = 'val'
    img_transformed, boxes, labels  = transform(img,phase,anno_info_list[:,:4],anno_info_list[:,4])
    print(img_transformed.shape)
    cv2.imshow("image test",cv2.cvtColor(img_transformed,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

    
