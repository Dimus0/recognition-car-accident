# yolov8_like.py
import math
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.ops import nms

# ----------------------------
# Utility modules
# ----------------------------
def autopad(k, p=None):  # pad to 'same'
    if p is None:
        p = k // 2
    return p

class Conv(nn.Module):
    """Standard conv + BN + SiLU"""
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C3(nn.Module):
    """CSP-ish block with 3 convs (bottleneck)"""
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_) for _ in range(n)])
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, 1)
        self.conv2 = Conv(c2, c2, 3, 1)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add else y

class SPP(nn.Module):
    """Spatial Pyramid Pooling"""
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x//2) for x in k])
    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

# ----------------------------
# YOLO-like backbone + neck + head
# ----------------------------
class YOLOBackbone(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256, 512]):
        super().__init__()
        self.stem = Conv(3, channels[0], 3, 1)
        # downsample blocks
        self.layer1 = nn.Sequential(Conv(channels[0], channels[1], 3, 2), C3(channels[1], channels[1], n=1))
        self.layer2 = nn.Sequential(Conv(channels[1], channels[2], 3, 2), C3(channels[2], channels[2], n=3))
        self.layer3 = nn.Sequential(Conv(channels[2], channels[3], 3, 2), C3(channels[3], channels[3], n=3))
        self.layer4 = nn.Sequential(Conv(channels[3], channels[4], 3, 2), C3(channels[4], channels[4], n=1))
        self.spp = SPP(channels[4], channels[4])
    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)  # small feature
        x2 = self.layer2(x1) # medium
        x3 = self.layer3(x2) # large
        x4 = self.layer4(x3) # x4
        x4 = self.spp(x4)
        return x2, x3, x4  # return feature maps for FPN/PAN

class YOLOPAN(nn.Module):
    """Simple top-down then bottom-up (PAnet-like)"""
    def __init__(self, channels=[128, 256, 512], out_channels=128):
        super().__init__()
        c2, c3, c4 = channels
        self.up1 = Conv(c4, c3//2, 1, 1)
        self.conv_for_merge1 = C3(c3, c3, n=1)
        self.up2 = Conv(c3, c2//2, 1, 1)
        self.conv_for_merge2 = C3(c2, c2, n=1)
        # bottom-up
        self.down1 = Conv(c2, c3//2, 3, 2)
        self.conv_down1 = C3(c3, c3, n=1)
        self.down2 = Conv(c3, c4//2, 3, 2)
        self.conv_down2 = C3(c4, c4, n=1)
        # output convs per detection scale
        self.out_small = Conv(c2, out_channels, 1, 1)  # high-res (small objects)
        self.out_medium = Conv(c3, out_channels, 1, 1)
        self.out_large = Conv(c4, out_channels, 1, 1)
    def forward(self, x2, x3, x4):
        # Top-down
        p4 = self.up1(x4)
        p3 = torch.cat([p4, x3], dim=1)
        p3 = self.conv_for_merge1(p3)
        p3_up = self.up2(p3)
        p2 = torch.cat([p3_up, x2], dim=1)
        p2 = self.conv_for_merge2(p2)
        # Bottom-up
        n2 = self.down1(p2)
        n3 = torch.cat([n2, p3], dim=1)
        n3 = self.conv_down1(n3)
        n3_down = self.down2(n3)
        n4 = torch.cat([n3_down, x4], dim=1)
        n4 = self.conv_down2(n4)
        return self.out_small(p2), self.out_medium(n3), self.out_large(n4)

class DetectHead(nn.Module):
    """
    Anchor-free detect head:
    Predicts (tx, ty, tw, th, obj, class_probs)
    """
    def __init__(self, in_channels, num_classes, width=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.conv = Conv(in_channels, in_channels, 3, 1)
        # output: 4 bbox + obj + classes
        self.out = nn.Conv2d(in_channels, (4 + 1 + num_classes), 1, 1, 0)
    def forward(self, x):
        x = self.conv(x)
        return self.out(x)

class YOLOv8Like(nn.Module):
    def __init__(self, num_classes=1, channels=[32,64,128,256,512], head_channels=128):
        super().__init__()
        self.backbone = YOLOBackbone(channels=channels)
        self.neck = YOLOPAN(channels=[channels[2], channels[3], channels[4]], out_channels=head_channels)
        self.head_s = DetectHead(head_channels, num_classes)
        self.head_m = DetectHead(head_channels, num_classes)
        self.head_l = DetectHead(head_channels, num_classes)
        self.num_classes = num_classes
    def forward(self, x):
        x2, x3, x4 = self.backbone(x)
        p2, p3, p4 = self.neck(x2, x3, x4)  # small, medium, large features
        out_s = self.head_s(p2)
        out_m = self.head_m(p3)
        out_l = self.head_l(p4)
        return [out_s, out_m, out_l]

# ----------------------------
# Losses & matching utilities
# ----------------------------
def bbox_iou(box1, box2, eps=1e-7):
    # boxes: [x1,y1,x2,y2]
    inter = (torch.min(box1[...,2], box2[...,2]) - torch.max(box1[...,0], box2[...,0])).clamp(0) * \
            (torch.min(box1[...,3], box2[...,3]) - torch.max(box1[...,1], box2[...,1])).clamp(0)
    w1 = (box1[...,2] - box1[...,0]).clamp(min=0)
    h1 = (box1[...,3] - box1[...,1]).clamp(min=0)
    w2 = (box2[...,2] - box2[...,0]).clamp(min=0)
    h2 = (box2[...,3] - box2[...,1]).clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter + eps
    return inter / union

def ciou_loss(pred_boxes, target_boxes):
    # both [N,4] x1y1x2y2
    iou = bbox_iou(pred_boxes, target_boxes)
    # center distance
    x1 = (pred_boxes[:,0] + pred_boxes[:,2]) / 2
    y1 = (pred_boxes[:,1] + pred_boxes[:,3]) / 2
    x2 = (target_boxes[:,0] + target_boxes[:,2]) / 2
    y2 = (target_boxes[:,1] + target_boxes[:,3]) / 2
    center_dist = (x1 - x2)**2 + (y1 - y2)**2
    # enclosure
    enclose_x1 = torch.min(pred_boxes[:,0], target_boxes[:,0])
    enclose_y1 = torch.min(pred_boxes[:,1], target_boxes[:,1])
    enclose_x2 = torch.max(pred_boxes[:,2], target_boxes[:,2])
    enclose_y2 = torch.max(pred_boxes[:,3], target_boxes[:,3])
    c2 = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 + 1e-7
    u = center_dist / c2
    # aspect ratio term (v)
    w1 = (pred_boxes[:,2] - pred_boxes[:,0]).clamp(min=1e-7)
    h1 = (pred_boxes[:,3] - pred_boxes[:,1]).clamp(min=1e-7)
    w2 = (target_boxes[:,2] - target_boxes[:,0]).clamp(min=1e-7)
    h2 = (target_boxes[:,3] - target_boxes[:,1]).clamp(min=1e-7)
    v = (4 / (math.pi**2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)
    ciou = iou - (u + alpha * v)
    return 1 - ciou  # loss

# ----------------------------
# Dataset
# ----------------------------
class YOLODataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, img_size=640, transforms=None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_files = sorted([p for p in self.images_dir.glob('*') if p.suffix.lower() in ('.jpg','.png','.jpeg')])
        self.img_size = img_size
        self.transforms = transforms if transforms is not None else default_transforms(img_size)
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.labels_dir / (img_path.stem + '.txt')
        img = Image.open(img_path).convert('RGB')
        w0, h0 = img.size
        img = self.transforms(img)  # tensor [3,H,W], normalized [0..1]
        boxes = []
        classes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.read().strip().splitlines():
                    if not line:
                        continue
                    parts = line.split()
                    cls = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:5])
                    # convert normalized to pixel coords (x1,y1,x2,y2)
                    x1 = (x_c - bw/2) * self.img_size
                    y1 = (y_c - bh/2) * self.img_size
                    x2 = (x_c + bw/2) * self.img_size
                    y2 = (y_c + bh/2) * self.img_size
                    boxes.append([x1, y1, x2, y2])
                    classes.append(cls)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,), dtype=torch.long)
        return img, {'boxes': boxes, 'labels': classes, 'orig_size': (w0, h0), 'img_path': str(img_path)}

def default_transforms(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),  # converts to [0..1]
    ])

# ----------------------------
# Target assignment (simple)
# ----------------------------
def build_targets(preds, targets, strides, device, num_classes, img_size):
    """
    preds: list of feature maps outputs (tensor shapes vary)
    targets: list of target dicts per image (boxes pixel coords)
    We'll implement a simple grid-based assignment: each gt assigned to the cell at its center at best scale by bbox area.
    Returns per-scale target tensors: obj_mask, box_target, class_target
    """
    batch_size = len(targets)
    num_scales = len(preds)
    toutputs = []
    for i, p in enumerate(preds):
        _, _, H, W = p.shape
        stride = img_size / H
        obj_mask = torch.zeros((batch_size, 1, H, W), device=device)
        box_target = torch.zeros((batch_size, 4, H, W), device=device)
        class_target = torch.zeros((batch_size, num_classes, H, W), device=device)
        toutputs.append((obj_mask, box_target, class_target, stride))
    # For each image and each gt, assign to a single best scale
    for b_i, t in enumerate(targets):
        boxes = t['boxes']  # pixel coordinates
        labels = t['labels']
        for j in range(boxes.shape[0]):
            x1,y1,x2,y2 = boxes[j]
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            # decide scale by area thresholds (simple)
            # small: area < (img_size/32)^2 etc. we map to scales by H size
            best_scale = 0
            # choose scale whose stride best matches bbox size
            min_diff = float('inf')
            for si, (_,_,_, stride) in enumerate(toutputs):
                # expected pixel span at that scale ~ stride * some factor => we measure diff from stride*bw_cross?
                diff = abs(math.log(bw + 1e-6) - math.log(stride + 1e-6))
                if diff < min_diff:
                    min_diff = diff
                    best_scale = si
            obj_mask, box_target, class_target, stride = toutputs[best_scale]
            H = obj_mask.shape[2]; W = obj_mask.shape[3]
            gx = xc / stride
            gy = yc / stride
            ix = int(gx.clamp(0, W-1))
            iy = int(gy.clamp(0, H-1))
            obj_mask[b_i, 0, iy, ix] = 1.0
            # encode box target as cx,cy,w,h relative to cell
            box_target[b_i, 0, iy, ix] = gx - ix
            box_target[b_i, 1, iy, ix] = gy - iy
            box_target[b_i, 2, iy, ix] = math.log(bw / img_size + 1e-8)
            box_target[b_i, 3, iy, ix] = math.log(bh / img_size + 1e-8)
            class_onehot = torch.zeros(num_classes, device=device)
            class_onehot[labels[j]] = 1.0
            class_target[b_i, :, iy, ix] = class_onehot
    return toutputs

# ----------------------------
# Training step
# ----------------------------
def compute_loss(preds, targets, device, num_classes, img_size):
    # preds: list of outputs per scale: [B, C, H, W]
    # targets: list of dicts (boxes pixel coords)
    bs = preds[0].shape[0]
    strides = []
    for p in preds:
        _,_,H,W = p.shape
        strides.append(img_size / H)
    # Build targets per scale
    toutputs = build_targets(preds, targets, strides, device, num_classes, img_size)
    loss_obj = 0.0
    loss_cls = 0.0
    loss_box = 0.0
    bceloss = nn.BCEWithLogitsLoss(reduction='sum')
    ce_loss = nn.BCEWithLogitsLoss(reduction='sum')  # for classes as one-hot
    for p, (obj_mask, box_target, class_target, stride) in zip(preds, toutputs):
        # reshape predictions
        B, C, H, W = p.shape
        p = p.view(B, -1, H, W)
        # channel order: [tx,ty,tw,th,obj,classes...]
        tx = p[:,0,:,:]
        ty = p[:,1,:,:]
        tw = p[:,2,:,:]
        th = p[:,3,:,:]
        tobj = p[:,4,:,:]
        tcls = p[:,5:5+num_classes,:,:]
        # object loss
        loss_obj += bceloss(tobj, obj_mask.squeeze(1)) / B
        # class loss
        loss_cls += ce_loss(tcls, class_target) / B
        # bbox loss (compute only where obj_mask==1)
        pos = obj_mask.squeeze(1) > 0
        if pos.sum() > 0:
            # pred boxes -> convert from tx,ty,tw,th to pixel coords
            # grid coords
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            grid_x = grid_x.unsqueeze(0).expand(B, -1, -1).float()
            grid_y = grid_y.unsqueeze(0).expand(B, -1, -1).float()
            pred_cx = (torch.sigmoid(tx) + grid_x) * stride
            pred_cy = (torch.sigmoid(ty) + grid_y) * stride
            pred_w = torch.exp(tw) * img_size
            pred_h = torch.exp(th) * img_size
            pred_x1 = pred_cx - pred_w/2
            pred_y1 = pred_cy - pred_h/2
            pred_x2 = pred_cx + pred_w/2
            pred_y2 = pred_cy + pred_h/2
            pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
            # collect target boxes in same shape
            # decode box_target
            tg_tx = box_target[:,0,:,:]
            tg_ty = box_target[:,1,:,:]
            tg_tw = box_target[:,2,:,:]
            tg_th = box_target[:,3,:,:]
            tg_cx = (tg_tx + grid_x) * stride
            tg_cy = (tg_ty + grid_y) * stride
            tg_w = torch.exp(tg_tw) * img_size
            tg_h = torch.exp(tg_th) * img_size
            tgt_x1 = tg_cx - tg_w/2
            tgt_y1 = tg_cy - tg_h/2
            tgt_x2 = tg_cx + tg_w/2
            tgt_y2 = tg_cy + tg_h/2
            tgt_boxes = torch.stack([tgt_x1, tgt_y1, tgt_x2, tgt_y2], dim=-1)
            # flatten where pos
            pred_boxes_pos = pred_boxes[pos].view(-1,4)
            tgt_boxes_pos = tgt_boxes[pos].view(-1,4)
            loss_box += ciou_loss(pred_boxes_pos, tgt_boxes_pos).sum() / B
    loss = loss_obj + loss_cls + loss_box
    return loss, {'obj': loss_obj.item(), 'cls': loss_cls.item(), 'box': loss_box.item()}

# ----------------------------
# Train / Eval loops
# ----------------------------
def train_one_epoch(model, dataloader, optimizer, device, epoch, img_size, num_classes):
    model.train()
    total_loss = 0.0
    for i, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        # move targets values types
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)
        preds = model(imgs)  # list of outputs
        loss, loss_items = compute_loss(preds, targets, device, num_classes, img_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f"Epoch {epoch} Step {i}/{len(dataloader)} loss {loss.item():.4f} parts {loss_items}")
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, img_size, conf_thres=0.25, iou_thres=0.5):
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            outs = model(imgs)
            # decode outputs and apply NMS
            batch_boxes = []
            for b in range(imgs.size(0)):
                boxes_all = []
                scores_all = []
                labels_all = []
                for p in outs:
                    B,C,H,W = p.shape
                    out = p[b]  # [C,H,W]
                    # decode
                    tx = out[0,:,:]
                    ty = out[1,:,:]
                    tw = out[2,:,:]
                    th = out[3,:,:]
                    tobj = out[4,:,:]
                    tcls = out[5:,:,:]  # [num_classes, H, W]
                    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                    stride = img_size / H
                    cx = (torch.sigmoid(tx) + grid_x) * stride
                    cy = (torch.sigmoid(ty) + grid_y) * stride
                    w = torch.exp(tw) * img_size
                    h = torch.exp(th) * img_size
                    obj_score = torch.sigmoid(tobj)
                    cls_prob = torch.sigmoid(tcls)  # [num_classes,H,W]
                    # compute final scores per cell per class
                    for cls_idx in range(cls_prob.shape[0]):
                        score_map = obj_score * cls_prob[cls_idx]
                        # threshold
                        mask = score_map > conf_thres
                        if mask.sum() == 0:
                            continue
                        ys, xs = torch.where(mask)
                        for yx in range(len(ys)):
                            yy = ys[yx].item(); xx = xs[yx].item()
                            sc = score_map[yy, xx].item()
                            bx = cx[yy, xx].item()
                            by = cy[yy, xx].item()
                            bw = w[yy, xx].item()
                            bh = h[yy, xx].item()
                            x1 = bx - bw/2; y1 = by - bh/2; x2 = bx + bw/2; y2 = by + bh/2
                            boxes_all.append([x1,y1,x2,y2])
                            scores_all.append(sc)
                            labels_all.append(cls_idx)
                if len(boxes_all) == 0:
                    batch_boxes.append([])
                    continue
                boxes_t = torch.tensor(boxes_all, device=device)
                scores_t = torch.tensor(scores_all, device=device)
                keep = nms(boxes_t, scores_t, iou_thres)
                kept = keep.cpu().numpy().tolist()
                detections = []
                for k in kept:
                    detections.append({
                        'box': boxes_all[k],
                        'score': float(scores_all[k]),
                        'label': int(labels_all[k])
                    })
                batch_boxes.append(detections)
            results.extend(batch_boxes)
    return results

# ----------------------------
# Example train runner
# ----------------------------
def train():
    # Hyperparams
    num_classes = 1  # vehicle / crash class — set as needed
    img_size = 640
    batch_size = 4
    epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv8Like(num_classes=num_classes).to(device)

    dataset_train = YOLODataset('dataset/images/train', 'dataset/labels/train', img_size=img_size)
    dataset_val = YOLODataset('dataset/images/val', 'dataset/labels/val', img_size=img_size)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        # note: our dataloader returns tuple(imgs, targets) because of custom collate
        avg_loss = train_one_epoch(model, loader_train, optimizer, device, epoch, img_size, num_classes)
        print(f"Epoch {epoch} avg loss {avg_loss:.4f}")
        scheduler.step()
        # evaluate every epoch or few epochs
        preds = evaluate(model, loader_val, device, img_size)
        print(f"Validation sample preds (first batch): {preds[:2]}")
        # optionally save
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pt')

if __name__ == '__main__':
    # Run training if executed directly
    train()
