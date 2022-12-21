# taken from https://github.com/optapy/optapy-quickstarts

from dataclasses import dataclass
import datetime
from functools import reduce
import random
import itertools

# Used to test for intersection of 2 arrays of timeslots
def intersection(a, b):
    return (bool(set(a) & set(b)))


# Assign id's based on index in list
def assign_ids(list):
    for i, l in enumerate(list):
        l.id = i

    return list


@dataclass
class Teacher:
    id: int
    name: str

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return self.id


@dataclass
class StudentGroup:
    id: int
    name: str

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return self.id


@dataclass
class Room:
    id: int
    name: str

    def __hash__(self):
        return self.id

    def __init__(self, name):
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

    # are these 2 timeslots consecutive - connected at the start/end ?
    def is_consecutive(self, other_timeslot):
        if other_timeslot.day_of_week != self.day_of_week:
            return False

        return self.start_time == other_timeslot.end_time or self.end_time == other_timeslot.start_time

    def __hash__(self):
        return self.id

    def __init__(self, day_of_week, start_time, end_time):
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

    def __init__(self, subject, teacher, student_group, n_timeslots):
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


# Group of lessons that must occupy the same timeslot
# this is for Electives
@dataclass
class LessonGroup:
    id: int
    lessons: list[Lesson]
    label: str

    def __init__(self, lessons, label):
        self.lessons = lessons
        self.label = label



def format_list(a_list):
    return ',\n'.join(map(str, a_list))


@dataclass
class Timetable:
    timeslots: list[Timeslot]
    rooms: list[Room]
    lessons: list[Lesson]
    teachers: list[Teacher]
    student_groups: list[StudentGroup]
    lesson_groups: list[LessonGroup]

    def __init__(self, timeslots, rooms, lessons, teachers, student_groups, lesson_groups=[]):
        self.timeslots = timeslots
        self.rooms = rooms
        self.lessons = lessons
        self.teachers = teachers
        self.student_groups = student_groups
        self.lesson_groups = lesson_groups

    def __str__(self):
        return (
            f"Timetable("
            f"timeslots={format_list(self.timeslots)},\n"
            f"rooms={format_list(self.rooms)},\n"
            f"lessons={format_list(self.lessons)},\n"
            f"score={str(self.score.toString()) if self.score is not None else 'None'}"
            f")"
        )

    def clear(self):
        for lesson in self.lessons:
            lesson.set_room(None)
            lesson.set_timeslots([])

    """
    def randomize_layout(self):
        "" " Make it all random, shuffle everything "" "
        for lesson in self.lessons:
            room_idx = random.randint(0, len(self.rooms) - 1)
            timeslot_idx = random.randint(0, len(self.timeslots) - 1)
            lesson.set_room(self.rooms[room_idx])
            lesson.set_timeslot(self.timeslots[timeslot_idx])
    """

    
    def find_consecutive_timeslots(self, n_timeslots, timeslots):
        """ Given a list of timeslots, find N in a row that are connected without breaks or locked """
        len_timeslots_list = len(timeslots)
        for ti in range(len_timeslots_list):
            t = timeslots[ti]
            if t.locked:
                continue

            r = [t]
            if len(r) == n_timeslots:
                return r

            for ni in range(ti + 1, len_timeslots_list):
                next_t = timeslots[ni]
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

    def find_consecutive_timeslots_reverse(self, n_timeslots, timeslots):
        """ Same as find_consecutive_timeslots but searches in the reverse direction """
        return self.find_consecutive_timeslots(n_timeslots, timeslots) # just do this for now....


    def find_free_timeslots_for_room(self, room):
        """ Given a room, find any vacant timeslots """
        timeslots = self.timeslots
        room_lessons = list(filter(lambda l: l.room == room, self.lessons))

        room_timeslots = []
        for lesson in room_lessons:
            room_timeslots += lesson.timeslots

        free_timeslots = list(filter(lambda t: t not in room_timeslots, timeslots))
        return free_timeslots


    def ordered_layout(self):
        """ Do a simple layout where each lesson is just placed down in order """
        self.clear()

        # TODO: handle lesson groups to ensure electives(lesson groups) all get the same timeslot just with different rooms
        for lesson in self.lessons:
            n_timeslots = lesson.n_timeslots
            found = False
            for ri, room in enumerate(self.rooms):
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
    rooms = timetable.rooms
    lessons = timetable.lessons
    timeslot_room_lesson_triple_list = list(map(lambda the_lesson: (the_lesson.timeslot, the_lesson.room, the_lesson),
                                                filter(lambda the_lesson:
                                                       the_lesson.timeslot is not None and
                                                       the_lesson.room is not None,
                                                lessons)))
    lesson_map = dict()
    for timeslot, room, lesson in timeslot_room_lesson_triple_list:
        if timeslot in lesson_map:
            if room in lesson_map[timeslot]:
                lesson_map[timeslot][room].append(lesson)
            else:
                lesson_map[timeslot][room] = [lesson]
        else:
            lesson_map[timeslot] = {room: [lesson]}

    print("|" + ("------------|" * (len(rooms) + 1)))
    print(reduce(lambda a, b: a + b + " | ",
                 map(lambda the_room: "{:<10}".format(the_room.name)[0:10], rooms),
                 "|            | "))
    print("|" + ("------------|" * (len(rooms) + 1)))
    for timeslot in timetable.timeslots:
        cell_list = list(map(lambda the_room: lesson_map.get(timeslot, {}).get(the_room, []),
                             rooms))
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
        print("|" + ("------------|" * (len(rooms) + 1)))
    unassigned_lessons = list(
        filter(lambda unassigned_lesson: unassigned_lesson.timeslot is None or unassigned_lesson.room is None,
               lessons))
    if len(unassigned_lessons) > 0:
        print()
        print("Unassigned lessons")
        for lesson in unassigned_lessons:
            print(" " + lesson.subject + " - " + lesson.teacher + " - " + lesson.student_group)

