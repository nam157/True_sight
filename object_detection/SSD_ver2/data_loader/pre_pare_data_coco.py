import torch
from torchvision.datasets import CocoDetection
from torch.utils.data.dataloader import default_collate
import os
from data_loader.pre_pare_data_coco import generate_dboxes,SSDTransformer
from torch.utils.data import DataLoader


class CocoDataset(CocoDetection):
    '''
    *root: đường dẫn data
    *year: năm của tập dataset,ex: coco2017
    *mode: train or vaildation
    *transform: processing data
    '''
    def __init__(self, root, year, mode, transform=None):
        annFile = os.path.join(root, "annotations", "instances_{}{}.json".format(mode, year))
        root = os.path.join(root, "{}{}".format(mode, year))
        super(CocoDataset, self).__init__(root, annFile)
        self._load_categories()
        self.transform = transform

    def _load_categories(self):

        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.label_map = {}
        self.label_info = {}
        counter = 1
        self.label_info[0] = "background"
        for c in categories:
            self.label_map[c["id"]] = counter
            self.label_info[counter] = c["name"]
            counter += 1

    def __getitem__(self, item):
        image, target = super(CocoDataset, self).__getitem__(item)
        width, height = image.size
        boxes = []
        labels = []
        if len(target) == 0:
            return None, None, None, None, None
        for annotation in target:
            bbox = annotation.get("bbox")
            boxes.append([bbox[0] / width, bbox[1] / height, (bbox[0] + bbox[2]) / width, (bbox[1] + bbox[3]) / height])
            labels.append(self.label_map[annotation.get("category_id")])
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        if self.transform is not None:
            image, (height, width), boxes, labels = self.transform(image, (height, width), boxes, labels)
        return image, target[0]["image_id"], (height, width), boxes, labels


def collate_fn(batch):
    '''
    Mình customize một chút với: Mỗi ảnh có nhiều nhãn(boxes,images) nên cần lấy ra nhiều nhãn tương ứng. 
    items[0] : images
    items[1] : target
    items[2] : (height,width)
    items[3] : boxes
    items[4] : labels
    '''
    items = list(zip(*batch)) 
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)]) #(batch_size,3,300,300)
    items[1] = list([i for i in items[1] if i])
    items[2] = list([i for i in items[2] if i])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)])
    return items

if __name__ == "__main__":
    data_path = '../input/coco-2017-dataset/coco2017/'

    dboxes = generate_dboxes(model="ssd")
    
    train_set = CocoDataset(data_path, 2017, "train", SSDTransformer(dboxes, (300, 300), val=False))
    train_loader = DataLoader(train_set, batch_size = 16, num_workers=6,shuffle=True)
    test_set = CocoDataset(data_path, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))
    test_loader = DataLoader(test_set,batch_size = 16,num_workers=6,shuffle=False)

    image, target, (height, width), boxes, labels = next(iter(train_loader))