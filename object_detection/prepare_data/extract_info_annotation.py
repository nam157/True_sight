from library import *
from make_data_path import make_data_list



class Annotation_xml(object):
    def __init__(self,classes):
        self.classes = classes

    def __call__(self,xml_path,w,h):
        
        # chua annotation cua anh
        ret = []

        # doc file xml 
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult  = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            #Chua thong tin bouding box
            bboxs = []
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin','ymin','xmax','ymax']
            for pt in pts:
                pixel = int(bbox.find(pt).text) - 1

                if pt == 'xmin' or pt == 'xmax':
                    pixel /= w  #ratio of width
                else: 
                    pixel /= h  #ratio of height
                
                bboxs.append(pixel)
            
            label_id = self.classes.index(name)
            bboxs.append(label_id)
            ret += [bboxs]
        return np.array(ret) #[xmin,ymin,xmax,ymax,label_id,...]


#### Test ham ####
if __name__ == "__main__":
    classes = ['aeroplane','bicycle','bird','boat',
            'bottle','bus','car','cat','chair','cow','diningtable','dog',
            'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    anno_xml = Annotation_xml(classes=classes)

    root_path = './VOC2007/'
    train_image_list,train_annotation_list,val_image_list,val_annotation_list = make_data_list(root_path)

    idx = 1
    img_file_path = val_image_list[idx]

    img = cv2.imread(img_file_path)

    h,w,c = img.shape
    print(h,w,c)

    extract_info = anno_xml.__call__(val_annotation_list[idx],w,h)
    print(extract_info)


    # cv2.imshow('test',img)
    # cv2.waitKey(10)
    





            