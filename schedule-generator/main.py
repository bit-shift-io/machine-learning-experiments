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
debug_initial_layout = False
if debug_initial_layout:
    timetable.start_layout()
    timetable_renderer = TimetableRenderer(render_mode='human')
    timetable_renderer.render(timetable, constraints)

render_mode = None #'human_fast' #'human_fast'
env = TimetableEnvV0(render_mode, timetable, constraints, max_episode_steps=200)
dnn = DNN(env.state_size(), env.action_size(), hidden_dim=1024, lr=0.0008)
checkpoint = dnn.load('model.pt')
n_trained_eps = 0
if checkpoint:
    n_trained_eps = checkpoint['n_episodes']
    print(f'Resuming training from episode #{n_trained_eps}')

# choose a training algorithm
#trainer = TA_QL_DoubleSoft(dnn, env, TAU=0.7)
trainer = TA_QL(dnn, env)

n_episodes = 100
trainer.train(n_episodes)
dnn.save('model.pt', {
    'n_episodes': n_trained_eps + n_episodes
})

beepy.beep(sound='ping')
trainer.plot() # this blocks

# run a few epochs in human mode for seeing how things look
env.renderer.render_mode = "human_fast"
trainer.epsilon = 0 # disable random actions
trainer.train_episode()
env.timetable.print()

print('Done')