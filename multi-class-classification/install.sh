#!/bin/bash

# Setup the project - download required dataset

mkdir data
cd data

curl -LO 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
