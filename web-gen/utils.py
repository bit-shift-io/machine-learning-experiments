import torch
import torchvision.ops.boxes as bops


# Convert boundary coordinates (x_min, y_min, w, h) with parent (w, h) into fractionnal scaling
# i.e. coords in % of parent
def to_fractional_scale(xy, parent_wh):
    return [xy[0] / parent_wh[0], xy[1] / parent_wh[1],  xy[2] / parent_wh[0], xy[3] / parent_wh[1]]

# Convert bounding boxes from boundary coordinates (x_min, y_min, w, h) to center-size coordinates (c_x, c_y, w, h)
def xy_to_cxcy(xy):
    return [xy[0] + (xy[2] / 2), xy[1] + (xy[3] / 2),  xy[2], xy[3]]

# convert center-size coordinates (c_x, c_y, w, h) to bbox (x1, y1, x2, y2)
def cxcy_to_box(cxcy):
    half_w = (cxcy[2] / 2)
    half_h = (cxcy[3] / 2)
    return [cxcy[0] - half_w, cxcy[1] - half_h, cxcy[0] + half_w, cxcy[1] + half_h]

# Intersection over union - a metric to determine how much 2 bounding boxes overlap
# Expects bounding boxes to be in center-size coords
def cxcy_to_iou(cxcy1, cxcy2):
    box1 = torch.tensor([cxcy_to_box(cxcy1)], dtype=torch.float)
    box2 = torch.tensor([cxcy_to_box(cxcy2)], dtype=torch.float)
    iou = bops.box_iou(box1, box2).item()
    return iou
