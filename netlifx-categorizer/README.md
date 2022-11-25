This experiment is to try to get some movies or dataset to self categorise.

My initial idea is to feed all movie id's in then have a seperate slot to put in 
a movie and its fields (tags, authors, etc..) and train it to match with the movie id.

This way when a new movie is released or an unknown movie is feed in, it tries to get
the closest movie id's it knows about. i.e. is recommends similar movies - that have similar tags, authors etc..)

Tutorial here which does something very similar: https://pub.towardsai.net/multi-label-text-classification-using-scikit-multilearn-case-study-with-stackoverflow-questions-768cb487ad12

https://wandb.ai/ayush-thakur/dl-question-bank/reports/A-Guide-to-Multi-Label-Classification-on-Keras--VmlldzoyMDgyMDU
https://pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/

There seems to good simple example here:
https://medium.com/geekculture/multilabel-not-multiclass-text-classification-using-keras-fa1c7992b195



A brief summary of the different output types: https://stackoverflow.com/a/49175655/500564

SO what I am doing with the 'tags' is 'multi-label' output, so here is a tutorial:
https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
