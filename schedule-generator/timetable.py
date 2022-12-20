# taken from https://github.com/optapy/optapy-quickstarts

from dataclasses import dataclass
import datetime
from functools import reduce
import random
import itertools

# Used to test for intersection of 2 arrays of timeslots
def intersection(a, b):
    return (bool(set(a) & set(b)))


@dataclass
class Teacher:
    id: int
    name: str

    def __hash__(self):
        return self.id


@dataclass
class StudentGroup:
    id: int
    name: str

    def __hash__(self):
        return self.id


@dataclass
class Room:
    id: int
    name: str

    def __hash__(self):
        return self.id

    def __init__(self, id, name):
        self.id = id
        self.name = name

    def __str__(self):
        return f"Room(id={self.id}, name={self.name})"


@dataclass
class Timeslot:
    id: int
    day_of_week: str
    start_time: datetime.time
    end_time: datetime.time
    label: str
    locked: bool

    def __hash__(self):
        return self.id

    def __init__(self, id, day_of_week, start_time, end_time):
        self.id = id
        self.day_of_week = day_of_week
        self.start_time = start_time
        self.end_time = end_time
        self.label = None
        self.locked = False

    def __str__(self):
        return (
                f"Timeslot("
                f"id={self.id}, "
                f"day_of_week={self.day_of_week}, "
                f"start_time={self.start_time}, "
                f"end_time={self.end_time})"
        )


@dataclass
class Lesson:
    id: int
    subject: str
    teacher: Teacher
    student_group: StudentGroup
    timeslots: list[Timeslot]
    n_timeslots: int # how many timeslots this lesson must occupy
    room: Room

    def __hash__(self):
        return self.id

    def __init__(self, id, subject, teacher, student_group, n_timeslots):
        self.id = id
        self.subject = subject
        self.teacher = teacher
        self.student_group = student_group
        self.timeslots = []
        self.n_timeslots = n_timeslots
        self.room = None
        self.constraints_fail = []
        self.constraints_pass = []

    def set_timeslots(self, new_timeslots):
        self.timeslots = new_timeslots

    def set_room(self, new_room):
        self.room = new_room

    def __str__(self):
        return (
            f"Lesson("
            f"id={self.id}, "
            f"timeslot={self.timeslot}, "
            f"room={self.room}, "
            f"teacher={self.teacher}, "
            f"subject={self.subject}, "
            f"student_group={self.student_group}"
            f")"
        )


def format_list(a_list):
    return ',\n'.join(map(str, a_list))


@dataclass
class Timetable:
    timeslot_list: list[Timeslot]
    room_list: list[Room]
    lesson_list: list[Lesson]
    teacher_list: list[Teacher]
    student_group_list: list[StudentGroup]

    def __init__(self, timeslot_list, room_list, lesson_list, teacher_list, student_group_list):
        self.timeslot_list = timeslot_list
        self.room_list = room_list
        self.lesson_list = lesson_list
        self.teacher_list = teacher_list
        self.student_group_list = student_group_list

    def __str__(self):
        return (
            f"Timetable("
            f"timeslot_list={format_list(self.timeslot_list)},\n"
            f"room_list={format_list(self.room_list)},\n"
            f"lesson_list={format_list(self.lesson_list)},\n"
            f"score={str(self.score.toString()) if self.score is not None else 'None'}"
            f")"
        )

    def clear(self):
        for lesson in self.lesson_list:
            lesson.set_room(None)
            lesson.set_timeslot(None)

    """
    def randomize_layout(self):
        "" " Make it all random, shuffle everything "" "
        for lesson in self.lesson_list:
            room_idx = random.randint(0, len(self.room_list) - 1)
            timeslot_idx = random.randint(0, len(self.timeslot_list) - 1)
            lesson.set_room(self.room_list[room_idx])
            lesson.set_timeslot(self.timeslot_list[timeslot_idx])
    """

    
    def find_consecutive_timeslots(self, n_timeslots, timeslot_list):
        """ Given a list of timeslots, find N in a row that are connected without breaks or locked """
        len_timeslots_list = len(timeslot_list)
        for ti in range(len_timeslots_list):
            t = timeslot_list[ti]
            if t.locked:
                continue

            r = [t]
            if len(r) == n_timeslots:
                return r

            for ni in range(ti + 1, len_timeslots_list):
                next_t = timeslot_list[ni]
                if next_t.locked:
                    break

                if t.day_of_week != next_t.day_of_week:
                    break

                if t.end_time != next_t.start_time:
                    break

                # so these 2 timeslots are consecutive!
                t = next_t
                r.append(t)

                if len(r) == n_timeslots:
                    return r

        return None

    def find_consecutive_timeslots_reverse(self, n_timeslots, timeslot_list):
        """ Same as find_consecutive_timeslots but searches in the reverse direction """
        return self.find_consecutive_timeslots(n_timeslots, timeslot_list) # just do this for now....


    def find_free_timeslots_for_room(self, room):
        """ Given a room, find any vacant timeslots """
        timeslots = self.timeslot_list
        room_lessons = list(filter(lambda l: l.room == room, self.lesson_list))

        room_timeslots = []
        for lesson in room_lessons:
            room_timeslots += lesson.timeslots

        free_timeslots = list(filter(lambda t: t not in room_timeslots, timeslots))
        return free_timeslots


    def ordered_layout(self):
        """ Do a simple layout where each lesson is just placed down in order """
        for lesson in self.lesson_list:
            n_timeslots = lesson.n_timeslots
            found = False
            for ri, room in enumerate(self.room_list):
                free_timeslots = self.find_free_timeslots_for_room(room)
                consecutive_timeslots = self.find_consecutive_timeslots(n_timeslots, free_timeslots)
                if consecutive_timeslots:
                    lesson.set_timeslots(consecutive_timeslots)
                    lesson.set_room(room)
                    found = True
                    break

            if not found:
                print("Not enough rooms to fit the lessons in. Try breaking down subjects into smaller lessons?")
                return
        
        return


    def print(self):
        #print_timetable(self) # TODO: support multiple timeslots
        pass




def print_timetable(timetable: Timetable):
    room_list = timetable.room_list
    lesson_list = timetable.lesson_list
    timeslot_room_lesson_triple_list = list(map(lambda the_lesson: (the_lesson.timeslot, the_lesson.room, the_lesson),
                                                filter(lambda the_lesson:
                                                       the_lesson.timeslot is not None and
                                                       the_lesson.room is not None,
                                                lesson_list)))
    lesson_map = dict()
    for timeslot, room, lesson in timeslot_room_lesson_triple_list:
        if timeslot in lesson_map:
            if room in lesson_map[timeslot]:
                lesson_map[timeslot][room].append(lesson)
            else:
                lesson_map[timeslot][room] = [lesson]
        else:
            lesson_map[timeslot] = {room: [lesson]}

    print("|" + ("------------|" * (len(room_list) + 1)))
    print(reduce(lambda a, b: a + b + " | ",
                 map(lambda the_room: "{:<10}".format(the_room.name)[0:10], room_list),
                 "|            | "))
    print("|" + ("------------|" * (len(room_list) + 1)))
    for timeslot in timetable.timeslot_list:
        cell_list = list(map(lambda the_room: lesson_map.get(timeslot, {}).get(the_room, []),
                             room_list))
        out = "| " + (timeslot.day_of_week[0:3] + " " + str(timeslot.start_time))[0:10] + " | "
        for cell in cell_list:
            if len(cell) == 0:
                out += "           | "
            else:
                out += "{:<10}".format(reduce(lambda a, b: a + "," + b,
                                              map(lambda assigned_lesson: assigned_lesson.subject,
                                                  cell)))[0:10] + " | "
        print(out)
        out = "|            | "
        for cell in cell_list:
            if len(cell) == 0:
                out += "           | "
            else:
                out += "{:<10}".format(reduce(lambda a, b: a + "," + b,
                                              map(lambda assigned_lesson: assigned_lesson.teacher.name,
                                                  cell)))[0:10] + " | "
        print(out)
        out = "|            | "
        for cell in cell_list:
            if len(cell) == 0:
                out += "           | "
            else:
                out += "{:<10}".format(reduce(lambda a, b: a + "," + b,
                                              map(lambda assigned_lesson: assigned_lesson.student_group.name,
                                                  cell)))[0:10] + " | "
        print(out)
        print("|" + ("------------|" * (len(room_list) + 1)))
    unassigned_lessons = list(
        filter(lambda unassigned_lesson: unassigned_lesson.timeslot is None or unassigned_lesson.room is None,
               lesson_list))
    if len(unassigned_lessons) > 0:
        print()
        print("Unassigned lessons")
        for lesson in unassigned_lessons:
            print(" " + lesson.subject + " - " + lesson.teacher + " - " + lesson.student_group)

