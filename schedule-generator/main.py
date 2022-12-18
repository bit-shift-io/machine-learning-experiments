from timetable_env_v0 import TimetableEnvV0
from timetable_env_v1 import TimetableEnvV1
from problem import generate_problem, constraint_list
from ta_ql_double_soft import TA_QL_DoubleSoft
from ta_ql import TA_QL
from dnn import DNN
import beepy

timetable = generate_problem()
constraints = constraint_list()
env = TimetableEnvV1(None, timetable, constraints, max_episode_steps=100)
dnn = DNN(env.state_size(), env.action_size(), hidden_dim=128, lr=0.0008)

# choose a training algorithm
#trainer = TA_QL_DoubleSoft(dnn, env, TAU=1.0)
trainer = TA_QL(dnn, env)

trainer.train(n_episodes=100)

beepy.beep(sound='ping')
trainer.plot() # this blocks

# run a few epochs in human mode for seeing how things look
env.renderer.render_mode = "human"
trainer.train(n_episodes=1)
env.timetable.print()