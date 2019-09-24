#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Train the model
#
# Helpful links: 
# https://medium.com/octavian-ai/finding-shortest-paths-with-graph-networks-807c5bbfc9c8

from Shared import *

def main():
    train_data = load_data('Train')
    test_data = load_data('Test')
    model = Model1(train_data, test_data)
    model.train()
    
    

if __name__ == "__main__":
    main()
