
from library import *
from transform import *
from make_data_path import *
from extract_info_annotation import *



class MyDataset(data.Dataset):
    """
    *img_list:  Danh sach duong dan cua image
    *anno_list: Danh sach chua annotations cua moi image
    *phase: la nhanh (train or val)
    *transform: Tien xu ly buc anh
    *anno_xml: trich xuat thong tin cua file annotations
    
    """
    def __init__(self,img_list,anno_list,phase,transform,anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self,idx):
        img,gt,h,w = self.pull_item(idx)
        return img,gt
    def pull_item(self,idx):
        img_file_path  = self.img_list[idx]
        img  = cv2.imread(img_file_path)
        h,w,c = img.shape

        #Lay ra thong tin annotations
        anno_file_path = self.anno_list[idx]
        ann_info = self.anno_xml(anno_file_path,w,h)

        #processing data
        img,boxes,labels = self.transform(img, self.phase, ann_info[:,:4], ann_info[:,4])

        #BGR -> RGB
        img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)
        
        #ground truth
        gt = np.hstack((boxes,np.expand_dims(labels,axis=1)))

        return img,gt,h,w


def my_collate_fn(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0]) #sample[0]=img
        targets.append(torch.FloatTensor(sample[1])) # sample[1]=annotation

    imgs = torch.stack(imgs, dim=0)    # (batch_size, 3, 300, 300)

    return imgs, targets


#Test
if __name__ == "__main__":
    root_path = './VOC2007/'
    train_image_list,train_annotation_list,val_image_list,val_annotation_list = make_data_list(root_path)

    #Transform
    color_mean = (104, 117, 123)
    input_size = 300
    transform = DataTransform(input_size,color_mean)

    #annotations xml
    classes = ['aeroplane','bicycle','bird','boat',
            'bottle','bus','car','cat','chair','cow','diningtable','dog',
            'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    anno_xml = Annotation_xml(classes=classes)


    train_dataset = MyDataset(train_image_list,train_annotation_list, phase='train',transform = transform, anno_xml=anno_xml)
    val_dataset = MyDataset(val_image_list,val_annotation_list, phase='val',transform = transform, anno_xml=anno_xml)


    batch_size = 4
    train_dataloader = data.DataLoader(train_dataset,batch_size = batch_size,shuffle=True,collate_fn=my_collate_fn)
    val_dataloader = data.DataLoader(val_dataset, batch_size = batch_size,shuffle=True,collate_fn=my_collate_fn)

    # print(next(iter(train_dataloader))[0])

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    images,targets = next(iter(dataloader_dict['train']))
    print(images.shape)
    print(images[1])
    print(targets[1].size())

    




