Reinforcement learning.

In cart pole I used this tutorial:
https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f
and got as far as 4. Deep Q Learning with Replay Memory

so want to take this further and explore the other steps (done in train1.py):
5. Double Deep Q Learning
6. Soft Updates


In train2.py I do some experiments:
- just keep the best scoring network (or best N networks, can we average them?) and clone it and progress from there.... will this stop us going backwards?
- experiment with custom reward - square it for example? crank score for a win
- somehow modify rewards in memory based on finale? back propogration of rewards
- drop out layer
- try traning on all memory?
- remove memory for really bad runs? or enhance memory from good runs?
- For each memory batch, rank it on how it trains the model and store it. Then for ever N episodes keep the best batches. To help remove useless memory yet provide good distribution over time.


