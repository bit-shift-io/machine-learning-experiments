#!/bin/bash

# Setup the project - download required dataset

mkdir data
cd data
curl -LO 'https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/goodbooks-10k.zip'
unzip goodbooks-10k.zip
rm goodbooks-10k.zip

python3 -m pip install --upgrade pip

pip3 install tensorflow-gpu
pip3 install tensorflow
pip3 install keras
pip3 install matplotlib
pip3 install pandas
pip3 install scikit-learn