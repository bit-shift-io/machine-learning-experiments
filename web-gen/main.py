import argparse
from PIL import Image
from transformer import Transformer
from config import *
from utils import *
from model_io import save, load
from model import CNN2
import json

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str)
args = parser.parse_args()

tr = Transformer(image_size=image_size)

image = Image.open(args.image)
X = tr.encode_input_image(image)

model = CNN2(image_size=tr.input_size(), out_features=tr.output_size()).to(device)

# load existing model
io_params = load(model_path, model, None, {
    'epoch': 0
})

model.eval() 

layout, first_child_size = model(X)

layout = tr.decode_layout_class(layout)

if layout == 'row':
    size = first_child_size[0]
elif layout == 'column':
    size = first_child_size[1]

dictionary ={ 
  "layout": layout, 
  "size": size,
} 

json_object = json.dumps(dictionary, indent = 4) 
print(json_object)
