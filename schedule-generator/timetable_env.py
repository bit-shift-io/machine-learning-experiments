# https://www.gymlibrary.dev/content/environment_creation/

from timetable import TimeTable
import gym
from gym import spaces
import pygame
import numpy as np
from problem import generate_problem

# really timetable env
class TimeTableEnv(gym.Env):
    timetable: TimeTable
    constraints: list

    def __init__(self, render_mode=None, timetable=generate_problem(), constraints=None):
        self.timetable = timetable
        self.constraints = constraints

        n_lessons = len(timetable.get_lesson_list())
        n_timeslots = len(timetable.get_timeslot_list())
        n_rooms = len(timetable.get_room_list())

        # For each Lesson we have 4 actions: "previous room", "next room", "previous timeslot", "next timeslot"
        self.action_space = spaces.Discrete(n_lessons * 4) # we cann try making this a MultiDiscrete in future to allow multiple changes in 1 step

        # Each lesson is 2 discreet spaces, where: which room it is inn, which timetabel slot it is in
        self.observation_space = spaces.Dict()
        for lesson in timetable.get_lesson_list():
            id = f"lesson_{lesson.id}"
            space = spaces.MultiDiscrete([n_rooms + 1, n_timeslots + 1]) # 0 represents nothing
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

        self.timetable.randomize()

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
        terminated = False
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        pass




from gym.envs.registration import register

register(
    id='TimeTable-v0',
    entry_point='timetable_env:TimeTableEnv',
    max_episode_steps=300,
)