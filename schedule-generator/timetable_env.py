# https://www.gymlibrary.dev/content/environment_creation/

from timetable import TimeTable
from timetable_renderer import TimetableRenderer
import gym
from gym import spaces
import pygame
import numpy as np
from problem import generate_problem, generate_problem_simple
import math

# really timetable env
class TimeTableEnv(gym.Env):
    renderer: TimetableRenderer
    timetable: TimeTable
    constraints: list

    def __init__(self, render_mode=None, timetable=None, constraints=None):
        #self.timetable = timetable
        #self.constraints = constraints

        self.timetable, self.constraints = generate_problem() #generate_problem_simple()
        self.renderer = TimetableRenderer(render_mode)

        # precompute some stuff
        self.n_lessons = len(self.timetable.get_lesson_list())
        self.n_timeslots = len(self.timetable.get_timeslot_list())
        self.n_rooms = len(self.timetable.get_room_list())

        self.max_hard_score, self.max_soft_score = self.constraints.max_score(self.timetable)

        # For each Lesson we have 6 actions (just first 4 for now): "previous room", "next room", "previous timeslot", "next timeslot", "remove from timetale", "add to timetable"
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_lessons * self.n_actions) # we cann try making this a MultiDiscrete in future to allow multiple changes in 1 step

        # Each lesson is 2 discreet spaces, where: which room it is inn, which timetabel slot it is in
        self.observation_space = spaces.Dict()
        for lesson in self.timetable.get_lesson_list():
            id = f"lesson_{lesson.id}"
            space = spaces.MultiDiscrete([self.n_rooms + 1, self.n_timeslots + 1]) # 0 represents nothing
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
            o[id] = np.array([room.get_id() if room else 0, timeslot.get_id() if timeslot else 0]) # maybe reserve 0 for invalid value
            #self.observation_space[id] = [room.get_id() - 1, timeslot.get_id() - 1]

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

        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, info

    def step(self, action):
         # Map the action (element of {0,1,2,3}) to the direction we walk in
        #direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        #self._agent_location = np.clip(
        #    self._agent_location + direction, 0, self.size - 1
        #)
        # An episode is done iff the agent has reached the target
        #terminated = np.array_equal(self._agent_location, self._target_location)

        # convert action into something usable
        lesson_idx = math.floor(action / self.n_actions)
        lesson_action = action % self.n_actions
        lesson = self.timetable.get_lesson_list()[lesson_idx]

        # "previous room", "next room", "previous timeslot", "next timeslot", "remove from timetale", "add to timetable"
        match lesson_action:
            case 0: # "previous room"
                idx = lesson.get_room().get_id() - 1
                idx -= 1
                idx = idx % self.n_rooms
                lesson.set_room(self.timetable.get_room_list()[idx])

            case 1: # "next room"
                idx = lesson.get_room().get_id() - 1
                idx += 1
                idx = idx % self.n_rooms
                lesson.set_room(self.timetable.get_room_list()[idx])

            case 2: #  "previous timeslot"
                idx = lesson.get_timeslot().get_id() - 1
                idx -= 1
                idx = idx % self.n_timeslots
                lesson.set_timeslot(self.timetable.get_timeslot_list()[idx])

            case 3: # "next timeslot"
                idx = lesson.get_timeslot().get_id() - 1
                idx += 1
                idx = idx % self.n_timeslots
                lesson.set_timeslot(self.timetable.get_timeslot_list()[idx])

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

            # crank the score!
            hard_score *= 100
            soft_score *= 100
            terminated = True

        reward = hard_score #1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.renderer.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.renderer.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        self.renderer.render(self.timetable)

    def close(self):
        self.renderer.close()




from gym.envs.registration import register

register(
    id='TimeTable-v0',
    entry_point='timetable_env:TimeTableEnv',
    max_episode_steps=100,
)