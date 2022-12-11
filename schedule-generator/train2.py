from timetable import TimeTable, Timeslot, Room, Lesson
from constraints import define_constraints
from timetable_env import TimeTableEnv
from datetime import time
from problem import generate_problem

#constraints = define_constraints()
timetable = generate_problem()
env = TimeTableEnv(timetable, None)
env.generate_random()
env.timetable.print()