from timetable_env import TimetableEnv
from problem import generate_problem
from train_algo import TrainAlgo
from dnn import DNN

timetable, constraints = generate_problem()
env = TimetableEnv(None, timetable, constraints) # "human"
dnn = DNN(env.state_size(), env.action_size(), hidden_dim=128, lr=0.0008)
trainer = TrainAlgo(dnn, env)
trainer.train(n_episodes=100)
trainer.plot()
env.timetable.print()