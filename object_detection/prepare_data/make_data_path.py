from library import *

def make_data_list(root_path):
    image_path_template = os.path.join(root_path,'JPEGImages','%s.jpg')
    annotation_path_template = os.path.join(root_path,'Annotations','%s.xml')

    train_id_names = os.path.join(root_path,'ImageSets/Main/train.txt')
    val_id_names = os.path.join(root_path,'ImageSets/Main/val.txt')

    train_image_list = []
    train_annotation_list = []

    for line in open(train_id_names):
        file_id = line.strip()
        image_path = (image_path_template % file_id)
        annotation_path = (annotation_path_template % file_id)
        
        train_image_list.append(image_path)
        train_annotation_list.append(annotation_path)
    
    val_image_list = []
    val_annotation_list = []

    for line in open(val_id_names):
        file_id = line.strip()
        image_path = (image_path_template % file_id)
        annotation_path = (annotation_path_template % file_id)
        
        val_image_list.append(image_path)
        val_annotation_list.append(annotation_path)

    return train_image_list,train_annotation_list,val_image_list,val_annotation_list

if __name__ == "__main__":
    root_path = './VOC2007/'
    train_image_list,train_annotation_list,val_image_list,val_annotation_list = make_data_list(root_path)
    
    print(train_image_list[0])
    print(len(train_image_list))
    
    