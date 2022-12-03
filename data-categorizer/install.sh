#!/bin/bash

# Setup the project - download required dataset

#pip3 install --user kaggle


python3 -m pip install --upgrade pip

pip3 install tensorflow-gpu
pip3 install tensorflow
pip3 install keras
pip3 install matplotlib
pip3 install pandas
pip3 install scikit-learn

mkdir data
cd data

# Extract data from https://www.kaggle.com/datasets/shivamb/netflix-shows?resource=download to 
# /data
#kaggle datasets download -d shivamb/netflix-shows


curl -LO 'https://s3-ap-southeast-2.amazonaws.com/content2.supermarketswap.com.au/products'
curl -LO 'https://s3-ap-southeast-2.amazonaws.com/content2.supermarketswap.com.au/recipes'

#kaggle datasets download -d shivamb/netflix-shows
#https://www.kaggle.com/datasets/shivamb/netflix-shows/download?datasetVersionNumber=5