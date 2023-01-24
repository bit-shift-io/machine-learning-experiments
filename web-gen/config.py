
model_path = 'model.pt'

# hyperparameters
batch_size = 32

keep_prob = 0.8
hidden_sz = 1024 #out_features * out_features # TODO: make this a hyper param?

image_size=[200, 200]

learning_rate = 0.001

train_pct = 0.8 #0.001 # should e about 0.8, reduce to lower to speed up training for testing only
training_epochs = 100 # should be abbout 100, reduce to speed up testing


node_classes = ['node', 'leaf']
node_classes_len = len(node_classes)

display_classes = ['row', 'column']
display_classes_len = len(display_classes)

layout_classes = ['row', 'column', 'grid']
layout_classes_len = len(display_classes)

bounds_len = 4
size_len = 2