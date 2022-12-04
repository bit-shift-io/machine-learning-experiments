Experimenting with creating a schedule from some constraints

eg. we have a T teachers that need to teach C classes
each T will have its own constraints, each C will also have its own constraints

can we generate a schedule that meets all the requirements?



lets say each day has 2 classes for simplicity: morning, afternoon (shift)
there are a few subjects we need to each 1 time per week: math, english, art
we have some teachers: bob, jane
 
we need to ensure teachers are assigned too a subject on a particular day
 
how will we handle other constraints such as:
   bob only teaches math
   janes only works 3 shifts per week
   bob wants to work 3 days in a row, but only 1 shift per day
   only 1 teacher can teach 1 subbject at once
 

It seems reinforcement learning is the best approach to use.

https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
https://keras-gym.readthedocs.io/en/stable/
see tutorial1.py






possible architectures:
   [day shift subject teacher] => yes/no to indicate we approve  
   [for each shift: yes/no indicating if the subject/teacher is availale] => [for each shift: [subject teacher]]