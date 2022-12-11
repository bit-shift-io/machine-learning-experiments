# taken from https://github.com/optapy/optapy-quickstarts

from dataclasses import dataclass
import datetime
from datetime import time
from functools import reduce

@dataclass
class Room:
    id: int
    name: str

    def __hash__(self):
        return self.id

    def __init__(self, id, name):
        self.id = id
        self.name = name

    def get_id(self):
        return self.id

    def __str__(self):
        return f"Room(id={self.id}, name={self.name})"


@dataclass
class Timeslot:
    id: int
    day_of_week: str
    start_time: datetime.time
    end_time: datetime.time

    def __hash__(self):
        return self.id

    def __init__(self, id, day_of_week, start_time, end_time):
        self.id = id
        self.day_of_week = day_of_week
        self.start_time = start_time
        self.end_time = end_time

    def get_id(self):
        return self.id

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
    teacher: str
    student_group: str
    timeslot: Timeslot
    room: Room

    def __hash__(self):
        return self.id

    def __init__(self, id, subject, teacher, student_group, timeslot=None, room=None):
        self.id = id
        self.subject = subject
        self.teacher = teacher
        self.student_group = student_group
        self.timeslot = timeslot
        self.room = room

    def get_id(self):
        return self.id

    def get_timeslot(self):
        return self.timeslot

    def set_timeslot(self, new_timeslot):
        self.timeslot = new_timeslot

    def get_room(self):
        return self.room

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
class TimeTable:
    timeslot_list: list[Timeslot]
    room_list: list[Room]
    lesson_list: list[Lesson]
    #score: HardSoftScore

    def __init__(self, timeslot_list, room_list, lesson_list, score=None):
        self.timeslot_list = timeslot_list
        self.room_list = room_list
        self.lesson_list = lesson_list
       #self.score = score

    def get_timeslot_list(self):
        return self.timeslot_list

    def get_room_list(self):
        return self.room_list

    def get_lesson_list(self):
        return self.lesson_list

    #def get_score(self):
    #    return self.score

    #def set_score(self, score):
    #    self.score = score
    
    def __str__(self):
        return (
            f"TimeTable("
            f"timeslot_list={format_list(self.timeslot_list)},\n"
            f"room_list={format_list(self.room_list)},\n"
            f"lesson_list={format_list(self.lesson_list)},\n"
            f"score={str(self.score.toString()) if self.score is not None else 'None'}"
            f")"
        )


def generate_problem():
    timeslot_list = [
        Timeslot(1, "MONDAY", time(hour=8, minute=30), time(hour=9, minute=30)),
        Timeslot(2, "MONDAY", time(hour=9, minute=30), time(hour=10, minute=30)),
        Timeslot(3, "MONDAY", time(hour=10, minute=30), time(hour=11, minute=30)),
        Timeslot(4, "MONDAY", time(hour=13, minute=30), time(hour=14, minute=30)),
        Timeslot(5, "MONDAY", time(hour=14, minute=30), time(hour=15, minute=30)),
        Timeslot(6, "TUESDAY", time(hour=8, minute=30), time(hour=9, minute=30)),
        Timeslot(7, "TUESDAY", time(hour=9, minute=30), time(hour=10, minute=30)),
        Timeslot(8, "TUESDAY", time(hour=10, minute=30), time(hour=11, minute=30)),
        Timeslot(9, "TUESDAY", time(hour=13, minute=30), time(hour=14, minute=30)),
        Timeslot(10, "TUESDAY", time(hour=14, minute=30), time(hour=15, minute=30)),
    ]
    room_list = [
        Room(1, "Room A"),
        Room(2, "Room B"),
        Room(3, "Room C")
    ]
    lesson_list = [
        Lesson(1, "Math", "A. Turing", "9th grade"),
        Lesson(2, "Math", "A. Turing", "9th grade"),
        Lesson(3, "Physics", "M. Curie", "9th grade"),
        Lesson(4, "Chemistry", "M. Curie", "9th grade"),
        Lesson(5, "Biology", "C. Darwin", "9th grade"),
        Lesson(6, "History", "I. Jones", "9th grade"),
        Lesson(7, "English", "I. Jones", "9th grade"),
        Lesson(8, "English", "I. Jones", "9th grade"),
        Lesson(9, "Spanish", "P. Cruz", "9th grade"),
        Lesson(10, "Spanish", "P. Cruz", "9th grade"),
        Lesson(11, "Math", "A. Turing", "10th grade"),
        Lesson(12, "Math", "A. Turing", "10th grade"),
        Lesson(13, "Math", "A. Turing", "10th grade"),
        Lesson(14, "Physics", "M. Curie", "10th grade"),
        Lesson(15, "Chemistry", "M. Curie", "10th grade"),
        Lesson(16, "French", "M. Curie", "10th grade"),
        Lesson(17, "Geography", "C. Darwin", "10th grade"),
        Lesson(18, "History", "I. Jones", "10th grade"),
        Lesson(19, "English", "P. Cruz", "10th grade"),
        Lesson(20, "Spanish", "P. Cruz", "10th grade"),
    ]
    lesson = lesson_list[0]
    lesson.set_timeslot(timeslot_list[0])
    lesson.set_room(room_list[0])

    return TimeTable(timeslot_list, room_list, lesson_list)



def print_timetable(timetable: TimeTable):
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
                                              map(lambda assigned_lesson: assigned_lesson.teacher,
                                                  cell)))[0:10] + " | "
        print(out)
        out = "|            | "
        for cell in cell_list:
            if len(cell) == 0:
                out += "           | "
            else:
                out += "{:<10}".format(reduce(lambda a, b: a + "," + b,
                                              map(lambda assigned_lesson: assigned_lesson.student_group,
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

