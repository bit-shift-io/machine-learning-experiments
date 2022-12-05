#!/bin/bash

python3 -m pip install --upgrade pip

pip3 install tensorflow-gpu
pip3 install tensorflow
pip3 install keras
pip3 install matplotlib
pip3 install pandas
pip3 install scikit-learn

pip3 install gym
pip3 install keras_gym
pip3 install pygame


# https://towardsdatascience.com/accelerated-tensorflow-model-training-on-intel-mac-gpus-aa6ee691f894
pip3 install tensorflow-macos 

pip3 install --user tensorflow-metal 
#pip3 uninstall tensorflow-metal


pip3 install torch torchvision
pip3 install ipython