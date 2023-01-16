
# Convert boundary coordinates (x_min, y_min, w, h) with parent (w, h) into fractionnal scaling
# i.e. coords in % of parent
def to_fractional_scale(xy, parent_wh):
    return [xy[0] / parent_wh[0], xy[1] / parent_wh[1],  xy[2] / parent_wh[0], xy[3] / parent_wh[1]]

# Convert bounding boxes from boundary coordinates (x_min, y_min, w, h) to center-size coordinates (c_x, c_y, w, h)
def xy_to_cxcy(xy):
    return [xy[0] + (xy[2] / 2), xy[1] + (xy[3] / 2),  xy[2], xy[3]]
