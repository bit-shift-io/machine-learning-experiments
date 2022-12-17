from timetable_env import TimetableEnv
from problem import generate_problem
from train_algo import TrainAlgo
from dnn import DNN
import beepy

timetable, constraints = generate_problem()
env = TimetableEnv(None, timetable, constraints, max_episode_steps=100) # "human"
dnn = DNN(env.state_size(), env.action_size(), hidden_dim=128, lr=0.0008)
trainer = TrainAlgo(dnn, env)
trainer.train(n_episodes=10)

beepy.beep(sound='ping')
trainer.plot() # this blocks

# run a few epochs in human mode for seeing how things look
env.renderer.render_mode = "human"
trainer.train(n_episodes=1)
env.timetable.print()