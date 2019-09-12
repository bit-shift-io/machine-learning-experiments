rat_mark = 0.5      # The current rat cell will be painteg by gray 0.5

visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.1