This experiment is to try to get some movies or dataset to self categorise.

My initial idea is to feed all movie id's in then have a seperate slot to put in 
a movie and its fields (tags, authors, etc..) and train it to match with the movie id.

This way when a new movie is released or an unknown movie is feed in, it tries to get
the closest movie id's it knows about. i.e. is recommends similar movies - that have similar tags, authors etc..)

So what I am doing with the 'tags' is 'multi-label' output, so here is a tutorial:
https://machinelearningmastery.com/multi-label-classification-with-deep-learning/

There is also a really good article here on ntext processing:
https://realpython.com/python-keras-text-classification/

In train1.py I'm just getting multi-label outputs working for the listed_in columnn. 
This works well.

In train2.py I add a bag-of-words (BoW) on the description columnn to also help find related movies.
This isn't working well, I thinkn the description is proving to much weight. How do we reduce it?
2 seperate models then I merge the weights manually?

In train3.py I use some SS real world data. I also experiment with Data Pipelines as mentioned in this tutorial: https://www.muratkarakaya.net/2022/11/part-d-preprocessing-text-with-tf-data.html?m=1

Future improvements: ingredients are sorted by most important to least, so being able to weight them would be great!
