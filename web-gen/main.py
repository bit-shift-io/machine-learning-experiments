import argparse
from PIL import Image
from transformer import Transformer
from config import *
from utils import *
from debug import *
from model_io import save, load
from model import CNN
import json
import torchvision.transforms.functional as FT

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str)
# TODO: add input clip-rect
args = parser.parse_args()

tr = Transformer(image_size=image_size)

image = Image.open(args.image)
X = tr.encode_input_image(image)

model = CNN(image_size=tr.input_size(), out_features=tr.output_size()) #.to(device)

# load existing model
io_params = load(model_path, model)

model.eval() 

print(model)

# visualize weights - first conv layer
plot_weights(model.conv, 0, single_channel=True)

with torch.no_grad():
    
  # recursively divide an image while it gives the same layout as expected_layout
  def divide(X, expected_layout=None, size_scale=1.0, depth=0):
    y_layout, y_first_child_size = model(torch.unsqueeze(X, dim=0)) # .to(device, non_blocking=True)

    #if depth == 0:
    show_data(X, y_first_child_size[0].cpu(), block=False, pause=2.0)

    layout = tr.decode_layout_class(y_layout[0].cpu())

    if expected_layout == None:
      expected_layout = layout
    else:
      if expected_layout != layout:
        return None, [size_scale]

    # a cheap way to stop the recursion.... really we should compute the actual pixels left and stop when nwe get to some threshold
    if depth > 3:
      return None, [size_scale]

    x_shape = X.shape # #color channels = 0, height = 1, width = 2

    if layout == 'row':
        size = y_first_child_size[0][0].item()

        # we probbably reached the end
        if size <= 0.0 or size >= 1.0:
          return layout, [size_scale]

        w = int(size * x_shape[2])
        x1 = FT.crop(X, 0, 0, x_shape[1], w)
        x2 = FT.crop(X, 0, w, x_shape[1], x_shape[2] - w)
    elif layout == 'column':
        size = y_first_child_size[0][1].item()

        # we probbably reached the end
        if size <= 0.0 or size >= 1.0:
          return layout, [size_scale]

        h = int(size * x_shape[1])
        x1 = FT.crop(X, 0, 0, h, x_shape[2])
        x2 = FT.crop(X, h, 0, x_shape[1] - h, x_shape[2])

    x1 = FT.resize(x1, tr.image_size)
    x2 = FT.resize(x2, tr.image_size)

    y1_layout, y1_sizes = divide(x1, expected_layout, size_scale * size, depth+1)
    y2_layout, y2_sizes = divide(x2, expected_layout, size_scale * (1.0 - size), depth+1)

    # concat sizes
    sizes = y1_sizes + y2_sizes

    return layout, sizes


  layout, sizes = divide(X)
  d = {
    'layout': layout,
    'sizes': sizes
  }
  json_object = json.dumps(d, indent = 4) 
  print(json_object)
