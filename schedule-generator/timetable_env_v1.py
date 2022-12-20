# https://www.gymlibrary.dev/content/environment_creation/

from timetable import Timetable
from timetable_renderer import TimetableRenderer
from timetable_env_v0 import TimetableEnvV0
import gym
from gym import spaces
import numpy as np
import math
from gym.spaces.utils import flatdim, flatten, flatten_space, unflatten

# really timetable env
class TimetableEnvV1(TimetableEnvV0):
    """ Extend v0 and add into the state where the constraint violations are """

    def __init__(self, render_mode=None, timetable=None, constraints=None, max_episode_steps=100):
        super().__init__(render_mode, timetable, constraints, max_episode_steps)

        # Each lesson is 3 discreet spaces, where: which room it is in, which timetable slot it is in
        # we now also include which constraints are being violated = 1 or 0 for pass
        self.observation_space = spaces.Dict()
        for lesson in self.timetable.lesson_list:
            id = f"lesson_{lesson.id}"

            #const_arr = np.empty(self.n_constraints)
            #const_arr.fill(1)

            # TODO: setup constraint voliaitions in lessons
            # TODO: also add in here Teacher and StudentGroups so AI can map issues so we might nnot even nneed constraints
            #       as currenntly the AI knows nnothing about teachers or student groups so can only guess!
            arr = np.array([self.n_rooms, self.n_teachers, self.n_student_groups])
            space = spaces.MultiDiscrete(arr)
            self.observation_space[id] = space

            # lessons can take up multiple timeslots
            id = f"lesson_{lesson.id}_timeslots"
            space = spaces.MultiBinary([self.n_timeslots]) 
            self.observation_space[id] = space


    def _get_obs(self):
        """Convert timetable to the oservation_space"""
        o = {}
        for lesson in self.timetable.lesson_list:
            id = f"lesson_{lesson.id}"

            # calc constraint state
            #const_arr = np.empty(self.n_constraints)
            #const_arr.fill(0)
            #for const in lesson.constraints_fail:
            #    const_arr[const.id] = 1

            arr = np.array([lesson.room.id, lesson.teacher.id, lesson.student_group.id])
            #arr = np.concatenate((arr, const_arr), axis=0)
            o[id] = arr

            # lessons can take up multiple timeslots
            id = f"lesson_{lesson.id}_timeslots"
            arr = np.zeros(self.n_timeslots)
            for t in lesson.timeslots:
                arr[t.id] = 1

            o[id] = arr
 
        return o




from gym.envs.registration import register

register(
    id='Timetable-v1',
    entry_point='timetable_env:TimetableEnvV1',
    max_episode_steps=100,
)