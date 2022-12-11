# https://www.gymlibrary.dev/content/environment_creation/

from timetable import TimeTable
import gym
from gym import spaces
import pygame
import numpy as np

# really timetable env
class TimeTableEnv(gym.Env):
    timetable: TimeTable
    constraints: list

    def __init__(self, timetable, constraints):
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
            id = lesson.id
            space = spaces.MultiDiscrete([n_rooms, n_timeslots])
            self.observation_space[id] = space


    def generate_random(self):
        self.timetable.randomize()
