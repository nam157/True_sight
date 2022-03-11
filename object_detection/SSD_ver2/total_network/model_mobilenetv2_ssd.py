import torch.nn as nn
from fastai import *
from torchvision.models import * 
from fastai.vision import *
# from torchvision.models.mobilenet import mobilenet_v2,InvertedResidual
from utils.default_box import DefBox
import torch
import numpy as np

feature_maps = {}
class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = mobilenet_v2(pretrained=True).features
        self.feature_extractor[14].conv[0][2].register_forward_hook(self.get_activation())

    def get_activation(self):
        def hook(self, input, output):
            feature_maps[0] = output.detach()

        return hook

    def forward(self, x):
        x = self.feature_extractor(x)
        return feature_maps[0], x


def SeperableConv2d(in_channels, out_channels, kernel_size=3):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                  groups=in_channels, padding=padding),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def StackedSeperableConv2d(ls_channels, multiplier):
    out_channels = 6 * multiplier
    layers = [SeperableConv2d(in_channels=in_channels, out_channels=out_channels) for in_channels in ls_channels]
    layers.append(nn.Conv2d(in_channels=ls_channels[-1], out_channels=out_channels, kernel_size=1))
    return nn.ModuleList(layers)

cfg = {
    "num_classes": 21, #VOC data include 20 class + 1 background class
    "input_size": 300, #SSD300
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4], # Tỷ lệ khung hình cho source1->source6`
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300], # Size of default box
    "min_size": [30, 60, 111, 162, 213, 264], # Size of default box
    "max_size": [60, 111, 162, 213, 264, 315], # Size of default box
    "aspect_ratios": [[2], [2,3], [2,3], [2,3], [2], [2]],
    "width_mul" : 1,
}
def decode(loc, defbox_list):
    """
    Tính các thông tin của dbox và offset và vẽ ra các bounding boxes
    Tính bbox từ offset information và default box
    *loc: [8732 4] [dx,dy,dw,dh]
    *dbox_list: size = [8732,4] (cx_d,cy_d,w_d,h_d)

    ****
    return:
    boxes: [xmin,ymin,xmax,ymax]
    """

    boxes = torch.cat((
        defbox_list[:, :2] + 0.1*loc[:, :2]*defbox_list[:, 2:],
        defbox_list[:, 2:]*torch.exp(loc[:,2:]*0.2)), dim=1)

    boxes[:, :2] -= boxes[:,2:]/2 #calculate xmin, ymin
    boxes[:, 2:] += boxes[:, :2] #calculate xmax, ymax

    return boxes

# non-maximum_supression
def nms(boxes, scores, overlap=0.45, top_k=200):
    """
    IOU: Tính overlap của 2 bounding boxes
    Khử đi các bounding boxes trùng nhau và không phải bounding boxes lớn nhất dự trên conf 
    *boxes: 8732 boxes (num_boxes,4)
    *score: mỗi bb có conf (num_boxes)
    *overlap: bao nhiêu phần trăm trùng nhau
    *top_k: mỗi object thì chúng ta lấy k boxes dự conf
    """ 
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()

    # boxes coordinate
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # area of boxes
    area = torch.mul(x2-x1, y2-y1)

    tmp_x1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    value, idx = scores.sort(0)
    idx = idx[-top_k:] # id của top 200 boxes có độ tự tin cao nhất

    while idx.numel() > 0:
        i = idx[-1] # id của box có độ tự tin cao nhất
        keep[count] = i
        count += 1

        if idx.size(0) == 1:
            break
        
        idx = idx[:-1] #id của boxes ngoại trừ box có độ tự tin cao nhất
        #information boxes
        torch.index_select(x1, 0, idx, out=tmp_x1) #x1
        torch.index_select(y1, 0, idx, out=tmp_y1) #y1
        torch.index_select(x2, 0, idx, out=tmp_x2) #x2
        torch.index_select(y2, 0, idx, out=tmp_y2) #y2

        tmp_x1 = torch.clamp(tmp_x1, min=x1[i]) # =x1[i] if tmp_x1 < x1[1]
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i]) # =y2[i] if tmp_y2 > y2[i]
        
        # chuyển về tensor có size mà index được giảm đi 1
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # overlap area
        inter = tmp_w*tmp_h
        others_area = torch.index_select(area, 0, idx) # diện tích của mỗi bbox
        union = area[i] + others_area - inter
        iou = inter/union
        idx = idx[iou.le(overlap)] # giữ lại id của box có overlap ít với bbox đang xét

    return keep, count

class Detect():
    """
    Trong một image ko chỉ có 1 object, và chúng ta cần đưa ra thông từ object (bboxes,labels)
    """
    def __init__(self, conf_thresh=0.01, top_k=200, nsm_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nsm_thresh

    def __call__(self, loc_data, conf_data, dbox_list):
    # def forward(self, loc_data, conf_data, dbox_list): #  Old version pytorch
        num_batch = loc_data.size(0) #batch_size (2,4,6,...32, 64, 128)
        num_dbox = loc_data.size(1) # 8732
        num_classe = conf_data.size(2) #21

        conf_data = self.softmax(conf_data) 
        # (batch_num, num_dbox, num_class) -> (batch_num, num_class, num_dbox)
        conf_preds = conf_data.transpose(2, 1)

        output = torch.zeros(num_batch, num_classe, self.top_k, 5)

        # xử lý từng bức ảnh trong một batch các bức ảnh
        for i in range(num_batch):
            # Tính bbox từ offset information và default box
            decode_boxes = decode(loc_data[i], dbox_list)

            # copy confidence score của ảnh thứ i
            conf_scores = conf_preds[i].clone()

            for cl in range(1, num_classe):
                c_mask = conf_scores[cl].gt(self.conf_thresh) # chỉ lấy những confidence > 0.01
                scores = conf_scores[cl][c_mask]
                if scores.nelement() == 0: #numel()
                    continue

                # đưa chiều về giống chiều của decode_boxes để tính toán
                l_mask = c_mask.unsqueeze(1).expand_as(decode_boxes) #(8732, 4)
                boxes = decode_boxes[l_mask].view(-1, 4) # (số box có độ tự tin lớn hơn > 0.01, 4)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SSDLite():
    def __init__(self, backbone = MobileNetV2(),num_classes = 21,width_mul = 1,phase = 'train'):
        super(SSDLite, self).__init__()
        self.feature_extractor = backbone
        self.phase = phase
        self.num_classes = num_classes

        self.additional_blocks = nn.ModuleList([
            InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
            InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
            InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
            InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
        ])
        header_channels = [round(576 * width_mul), 1280, 512, 256, 256, 64]
        self.loc = StackedSeperableConv2d(header_channels, 4)
        self.conf = StackedSeperableConv2d(header_channels, self.num_classes)

        dbox = DefBox(cfg)
        self.dbox_list = dbox.create_defbox()
        if phase == "inference":
            self.detect = Detect()
        self._initialize_weights()

    def forward(self,x):
        y,x =  self.feature_extractor(x)
        loc = list()
        conf = list()
        sources = [y,x]
        for l in self.additional_blocks:
            x = l(x)
            sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            # (batch_size, 4*aspect_ratio_num, featuremap_height, featuremap_width) -> (batch_size, featuremap_height, featuremap_width ,4*aspect_ratio_num)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1) #(batch_size, 34928) 4*8732
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1) #(batch_size, 8732*21)

        loc = loc.view(loc.size(0), -1, 4) #(batch_size, 8732, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes) #(batch_size, 8732, 21)

        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":
            with torch.no_grad():
                return self.detect(output[0], output[1], output[2])
        else:
            return output
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, np.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             n = m.weight.size(1)
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()
    def _initialize_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)




