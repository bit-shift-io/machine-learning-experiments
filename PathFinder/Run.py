#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Shared import *

pause_time = 0.3
max_iterations = 100

def main():
    dataset = DataSet()
    model = Model1(dataset)
    model.load_weights()
    
    # pick a random map to run on
    map = random.choice(dataset.maps)
    map_1d_shape = map.map_1d.shape
    map_2d_shape = map.map_2d.shape


    # setup a history of zeros to draw on
    history = np.array([])
    pos_history = np.array([])

    for h in range(0, model.history_count):
        history = np.append(history, np.zeros(map_1d_shape))
        pos_history = np.append(pos_history, (0, 0))

    # and push on the current position
    pos = (0, 0)
    current_location_image_vec = generate_location_image_vector(map_2d_shape, pos)
    history = np.append(history, current_location_image_vec)
    pos_history = np.append(pos_history, pos)

    # make shape useful
    history = history.reshape((-1, map_1d_shape[0]))

    # visualise
    end_pos = add_tuple(map_2d_shape, (-1, -1))

    i = 0
    while (pos != end_pos):
        action = model.predict(map, history)
        movement = action_direction_map[action]
        new_pos = add_tuple(pos, movement)

        pos = new_pos

        current_location_image_vec = generate_location_image_vector(map_2d_shape, pos)
        history = np.append(history, current_location_image_vec)
        pos_history = np.append(pos_history, pos)

        # make shape useful
        history = history.reshape((-1, map_1d_shape[0]))
        pos_history = pos_history.reshape((-1, 2))

        show(map, pos_history, pause_time)

        i += 1
        if (i >= max_iterations):
            print("Failed to reach end position")
            break; 
    

if __name__ == "__main__":
    main()
