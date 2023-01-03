from timetable_env_v0 import TimetableEnvV0
from timetable_renderer import TimetableRenderer
from problem import generate_problem_large, constraint_list
from ta_ql_double_soft import TA_QL_DoubleSoft
from ta_ql import TA_QL
from dnn import DNN
import beepy

timetable = generate_problem_large()
constraints = constraint_list()

# debug the initial layout of the timetable
debug_initial_layout = True
if debug_initial_layout:
    timetable.ordered_layout()
    timetable_renderer = TimetableRenderer(render_mode='human')
    timetable_renderer.render(timetable, constraints)

render_mode = 'human' #'human_fast'
env = TimetableEnvV0(render_mode, timetable, constraints, max_episode_steps=100)
dnn = DNN(env.state_size(), env.action_size(), hidden_dim=128, lr=0.0008)

# choose a training algorithm
#trainer = TA_QL_DoubleSoft(dnn, env, TAU=0.7)
trainer = TA_QL(dnn, env)

trainer.train(n_episodes=300)

beepy.beep(sound='ping')
trainer.plot() # this blocks

# run a few epochs in human mode for seeing how things look
env.renderer.render_mode = "human_fast"
trainer.epsilon = 0 # disable random actions
trainer.train_episode()
env.timetable.print()

print('Done')