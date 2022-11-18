#!/bin/bash

# Setup the project - download required dataset

mkdir data
cd data
curl -LO 'https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/goodbooks-10k.zip'
unzip goodbooks-10k.zip
rm goodbooks-10k.zip