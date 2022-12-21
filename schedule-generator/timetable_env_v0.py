# https://www.gymlibrary.dev/content/environment_creation/

from timetable import Timetable, intersection, assign_ids
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
        # swap to another room with the same timeslots
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
            other_lesson = other_lessons[0]
            if other_lesson.n_timeslots != lesson.n_timeslots:
                print("oh dear, we couldn't swap rooms!... need more work to handle swapping")
                return

            other_lesson.set_room(cur_room)

        lesson.set_room(next_room)


    def action_swap_timeslot(self, lesson, offset):
        # swap timeslots in same room
        # TODO: support multiple timeslots

        # keep it simple, swap the lessons in the timetable and just called ordered_layout again!
        lesson_id = lesson.id
        next_lesson_id = (lesson_id + offset) % len(self.timetable.lesson_list)

        # swap
        self.timetable.lesson_list[lesson_id], self.timetable.lesson_list[next_lesson_id] = self.timetable.lesson_list[next_lesson_id], self.timetable.lesson_list[lesson_id]

        # update id's
        assign_ids(self.timetable.lesson_list)

        self.timetable.ordered_layout()

        """
        if offset > 0:
            last_timeslot = lesson.timeslots[-1]
            at_end = (last_timeslot.id + 1) == len(self.timetable.timeslot_list)
            next_timeslot = self.timetable.timeslot_list[last_timeslot.id + 1] if not at_end else self.timetable.timeslot_list[0]
            is_consecutive = next_timeslot.is_consecutive(last_timeslot)
            next_lesson = next(filter(lambda l: intersection([next_timeslot], l.timeslots) and l.room == lesson.room, self.timetable.lesson_list), None)

            # next timeslot is consecutive and no lesson below us, so move down 1 timeslot
            if is_consecutive and not next_lesson:
                temp = lesson.timeslots + [next_timeslot]
                lesson.set_timeslots(temp[-lesson.n_timeslots:])

            # do a straight swap with the next lesson
            elif next_lesson and next_lesson.n_timeslots == lesson.n_timeslots:
                temp = lesson.timeslots
                lesson.set_timeslots(next_lesson.timeslots)
                next_lesson.set_timeslots(temp)

            # do some weird swap
            elif is_consecutive and next_lesson:
                temp = lesson.timeslots + next_lesson.timeslots
                lesson_timeslots = temp[-lesson.n_timeslots:]
                next_lesson_timeslots = temp[:next_lesson.n_timeslots]
                lesson.set_timeslots(lesson_timeslots)
                next_lesson.set_timeslots(next_lesson_timeslots)

            else:
                print("unhandled scenario")

        elif offset < 0:
            last_timeslot = lesson.timeslots[0]
            at_end = last_timeslot.id == 0
            next_timeslot = self.timetable.timeslot_list[last_timeslot.id - 1] if not at_end else self.timetable.timeslot_list[-1]
            is_consecutive = next_timeslot.is_consecutive(last_timeslot)
            next_lesson = next(filter(lambda l: intersection([next_timeslot], l.timeslots) and l.room == lesson.room, self.timetable.lesson_list), None)

            # next timeslot is consecutive and no lesson below us, so move up 1 timeslot
            if is_consecutive and not next_lesson:
                temp = [next_timeslot] + lesson.timeslots
                lesson.set_timeslots(temp[:lesson.n_timeslots])

            # do a straight swap with the next lesson
            elif next_lesson and next_lesson.n_timeslots == lesson.n_timeslots:
                temp = lesson.timeslots
                lesson.set_timeslots(next_lesson.timeslots)
                next_lesson.set_timeslots(temp)

            # do some weird swap
            elif is_consecutive and next_lesson:
                temp = next_lesson.timeslots + lesson.timeslots
                lesson_timeslots = temp[:lesson.n_timeslots]
                next_lesson_timeslots = temp[-next_lesson.n_timeslots:] 
                lesson.set_timeslots(lesson_timeslots)
                next_lesson.set_timeslots(next_lesson_timeslots)

            else:
                print("unhandled scenario")
        """

        return



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
        #if hard_score == self.max_hard_score:
        #    print("Solution found!")
        #    self.timetable.print()
        #    terminated = True

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