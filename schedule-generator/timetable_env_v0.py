# https://www.gymlibrary.dev/content/environment_creation/

from timetable import Timetable, intersection
from timetable_renderer import TimetableRenderer
import gym
from gym import spaces
import numpy as np
import math
from gym.spaces.utils import flatdim, flatten, flatten_space, unflatten

# really timetable env
class TimetableEnvV0(gym.Env):
    """ Timetable Environment bbasically turns the timetable problem into a game """

    renderer: TimetableRenderer
    timetable: Timetable
    constraints: list

    def state_size(self):
        # Number of states
        return flatten_space(self.observation_space).shape[0]

    def action_size(self):
        # Number of actions
        return self.action_space.n

    def __init__(self, render_mode=None, timetable=None, constraints=None, max_episode_steps=100):
        #self.timetable = timetable
        #self.constraints = constraints
        self.max_episode_steps = max_episode_steps
        self.renderer = TimetableRenderer(render_mode)
        self.timetable = timetable
        self.constraints = constraints

        # precompute some stuff
        self.n_constraints = len(self.constraints.constraints)
        self.n_lessons = len(self.timetable.lesson_list)
        self.n_timeslots = len(self.timetable.timeslot_list)
        self.n_rooms = len(self.timetable.room_list)
        self.n_teachers = len(self.timetable.teacher_list)
        self.n_student_groups = len(self.timetable.student_group_list)

        self.max_hard_score, self.max_soft_score = self.constraints.max_score(self.timetable)

        # For each Lesson we have 6 actions (just first 4 for now): "previous room", "next room", "previous timeslot", "next timeslot", "remove from timetale", "add to timetable"
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_lessons * self.n_actions) # we cann try making this a MultiDiscrete in future to allow multiple changes in 1 step

        # Each lesson is 2 discreet spaces, where: which room it is inn, which timetabel slot it is in
        self.observation_space = spaces.Dict()
        for lesson in self.timetable.lesson_list:
            id = f"lesson_{lesson.id}"
            space = spaces.MultiDiscrete([self.n_rooms, self.n_timeslots])
            self.observation_space[id] = space
            #test = space.sample()
            #print('lesson', id)


    def _get_obs(self):
        """Convert timetable to the oservation_space"""
        o = {}
        for lesson in self.timetable.get_lesson_list():
            id = f"lesson_{lesson.id}"
            room = lesson.get_room()
            timeslot = lesson.get_timeslot()
            o[id] = np.array([room.id, timeslot.id]) # maybe reserve 0 for invalid value
            #self.observation_space[id] = [room.id - 1, timeslot.id - 1]

        return o #return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #self.timetable.randomize_layout()
        self.timetable.ordered_layout()

        # score here if just for testing
        hard_score, soft_score = self.constraints.test(self.timetable)

        observation = self._get_obs()
        info = self._get_info()

        self.n_step = 0

        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, info


    def action_swap_room(self, lesson, offset):
        # TODO: support multiple timeslots

        idx = lesson.room.id
        idx += offset
        idx = idx % self.n_rooms

        cur_room = lesson.room
        next_room = self.timetable.room_list[idx]

        other_lessons = list(filter(lambda l: l.room == next_room and intersection(l.timeslots, lesson.timeslots), self.timetable.lesson_list))
        if (len(other_lessons) > 1):
            print("oh dear, we cant swap rooms! unless the two lessons are the same size together, or one of them is the same size as this lesson")
            return

        if (len(other_lessons) == 1):
            other_lessons[0].set_room(cur_room)

        lesson.set_room(next_room)


    def action_swap_timeslot(self, lesson, offset):
        # TODO: support multiple timeslots

        next_timeslots = None
        while True:
            if offset > 0:
                timeslot_list = list(filter(lambda t: t.id > lesson.timeslots[-1].id, self.timetable.timeslot_list))
                next_timeslots = self.timetable.find_consecutive_timeslots(lesson.n_timeslots, timeslot_list)
            else:
                timeslot_list = list(filter(lambda t: t.id < lesson.timeslots[0].id, self.timetable.timeslot_list))
                next_timeslots = self.timetable.find_consecutive_timeslots_reverse(lesson.n_timeslots, timeslot_list)

            if next_timeslots:
                # TODO: check if this holds a single lesson... for now assume it does
                break

        other_lessons = list(filter(lambda l: intersection(l.timeslots, next_timeslots) and l.room == lesson.room, self.timetable.lesson_list))
        if (len(other_lessons) > 1):
            print("oh dear, we cant swap timeslots! unless the two lessons are the same size together, or one of them is the same size as this lesson")
            return

        if (len(other_lessons) == 1):
            other_lessons[0].set_timeslots(lesson.timeslots)

        lesson.set_timeslots(next_timeslots)

        """
        idx = lesson.timeslot.id
        idx += offset
        idx = idx % self.n_timeslots

        cur_timeslot = lesson.timeslot
        next_timeslot = self.timetable.timeslot_list[idx]

        other_lesson = next(filter(lambda l: l.timeslot == next_timeslot and l.room == lesson.room, self.timetable.lesson_list), None)
        if other_lesson:
            other_lesson.set_timeslot(cur_timeslot)

        lesson.set_timeslot(next_timeslot)
        """


    def step(self, action):
         # Map the action (element of {0,1,2,3}) to the direction we walk in
        #direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        #self._agent_location = np.clip(
        #    self._agent_location + direction, 0, self.size - 1
        #)
        # An episode is done iff the agent has reached the target
        #terminated = np.array_equal(self._agent_location, self._target_location)

        self.n_step += 1

        # convert action into something usable
        lesson_idx = math.floor(action / self.n_actions)
        lesson_action = action % self.n_actions
        lesson = self.timetable.lesson_list[lesson_idx]

        # "previous room", "next room", "previous timeslot", "next timeslot", "remove from timetale", "add to timetable"
        match lesson_action:
            case 0: # "previous room"
                self.action_swap_room(lesson, -1)

            case 1: # "next room"
                self.action_swap_room(lesson, 1)

            case 2: #  "previous timeslot"
                self.action_swap_timeslot(lesson, -1)

            case 3: # "next timeslot"
                self.action_swap_timeslot(lesson, 1)

            #case 4: # "remove from timetable"
            #    pass # TODO:

            #case 5: # "add to timetable"
            #    pass # TODO:

            case _: # default
                print("Unknown lesson_action!")

        # Score the new timetable
        # TODO: handle soft score
        hard_score, soft_score = self.constraints.test(self.timetable)

        terminated = False
        if hard_score == self.max_hard_score:
            print("Solution found!")
            self.timetable.print()
            terminated = True

        reward = hard_score #1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.renderer.render_mode == "human":
            self._render_frame()

        # stop if we reach to many steps... failure
        truncated = False
        if self.n_step > self.max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.renderer.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        self.renderer.render(self.timetable, self.constraints)

    def close(self):
        self.renderer.close()




from gym.envs.registration import register

register(
    id='Timetable-v0',
    entry_point='timetable_env:TimetableEnvV0',
    max_episode_steps=100,
)