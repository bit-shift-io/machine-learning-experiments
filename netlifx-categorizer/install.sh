#!/bin/bash

# Setup the project - download required dataset

pip3 install --user kaggle

mkdir data
cd data

# Extract data from https://www.kaggle.com/datasets/shivamb/netflix-shows?resource=download to 
# /data
kaggle datasets download -d shivamb/netflix-shows


#curl -LO 'https://www.kaggle.com/datasets/shivamb/netflix-shows/download?datasetVersionNumber=5'
#unzip goodbooks-10k.zip
#rm goodbooks-10k.zip

#kaggle datasets download -d shivamb/netflix-shows
#https://www.kaggle.com/datasets/shivamb/netflix-shows/download?datasetVersionNumber=5