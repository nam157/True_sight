from library import *


device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
class Model():
    def __init__(self,model = SSD(backbone=ResNet()) ,checkpoint = torch.load(config.pretrained_model)):
        self.model = model
        self.checkpoint = checkpoint
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.dboxes = generate_dboxes()
        self.transformer = SSDTransformer(self.dboxes, (300, 300), val=True)
        self.encoder = Encoder(self.dboxes)
    def pre_pare(self,frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame, _, _, _ = self.transformer(frame, None, torch.zeros(1, 4), torch.zeros(1))
        frame = frame.to(device)
        with torch.no_grad():
            ploc, plabel = self.model(frame.unsqueeze(dim=0))
            result = self.encoder.decode_batch(ploc, plabel, config.nms_threshold, 20)[0]
            loc, label, prob = [r.cpu().numpy() for r in result]
            best = np.argwhere(prob > config.cls_threshold).squeeze(axis=1)
            loc = loc[best]
            label = label[best]
            prob = prob[best]
        
            return loc,label,prob 
    def draw_object_info(self,frame):
        loc,label,prob = self.pre_pare(frame)
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
                cv2.rectangle(frame, (xmin, ymin), (xmin + text_size[0], ymin + text_size[1] + 4), color,-1)
                cv2.putText(frame, category + " : %.2f" % pb,(xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,(255, 255, 255), 1)
        return frame,center_point,(xmax-xmin),category
















