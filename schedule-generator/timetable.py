# taken from https://github.com/optapy/optapy-quickstarts

from dataclasses import dataclass
import datetime
from functools import reduce
import random
import itertools

# Used to test for intersection of 2 arrays of timeslots
def is_intersection(a, b):
    return (bool(set(a) & set(b)))

# Get the inntersectionn of 2 arrays
def intersection(a, b):
    return set(a) & set(b)


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
class Timeslotable:
    timeslots: list[Timeslot]
    n_timeslots: int # how many timeslots this timeslotable must occupy

    def __init__(self, n_timeslots):
        self.timeslots = []
        self.n_timeslots = n_timeslots

    def set_timeslots(self, new_timeslots):
        self.timeslots = new_timeslots

    def clear(self):
        self.set_timeslots(None)

    def is_elective(self):
        return False


@dataclass
class Lesson(Timeslotable):
    id: int
    subject: str
    teacher: Teacher
    student_group: StudentGroup
    #timeslots: list[Timeslot]
    #n_timeslots: int # how many timeslots this lesson must occupy
    room: Room
    elective: "Elective" # https://stackoverflow.com/questions/55320236/does-python-evaluate-type-hinting-of-a-forward-reference

    # Is this lesson in the same elective group as the other lesson?
    def is_same_elective(self, other):
        if self.elective == None or other.elective == None:
            return False

        return self.elective == other.elective

    def __hash__(self):
        return self.id

    def __init__(self, subject, teacher, student_group, n_timeslots):
        super().__init__(n_timeslots)
        self.subject = subject
        self.teacher = teacher
        self.student_group = student_group
        self.room = None
        self.elective = None
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

    def clear(self):
        super().clear()
        self.set_room(None)


# Group of lessons that must occupy the same timeslot
@dataclass
class Elective(Timeslotable):
    id: int
    lessons: list[Lesson]
    label: str

    def __hash__(self):
        return self.id

    def __init__(self, lessons, label):
        self.lessons = lessons
        self.label = label

        for lesson in self.lessons:
            lesson.elective = self
            
        super().__init__(lessons[0].n_timeslots)

    # shortcut to create multiple Electives for a set of Lessons
    def zipLessons(lesson_arr, label):
        electives = []
        for lessons in zip(*lesson_arr):
            electives.append(Elective(list(lessons), label))

        return electives

    def set_timeslots(self, new_timeslots):
        super().set_timeslots(new_timeslots)
        for lesson in self.lessons:
            lesson.set_timeslots(new_timeslots)

    def clear(self):
        super().clear()
        for lesson in self.lessons:
            lesson.clear()

    def is_elective(self):
        return True



# flatten an array of timeslotables into a list of lessons
def get_lessons(timeslotables):
    lessons = []
    for timeslotable in timeslotables:
        if (timeslotable.is_elective()):
            lessons += timeslotable.lessons
        else:
            lessons += [timeslotable]

    return lessons


def format_list(a_list):
    return ',\n'.join(map(str, a_list))


@dataclass
class Timetable:
    timeslots: list[Timeslot]
    timeslotables: list[Timeslotable]
    lessons: list[Lesson]

    rooms: list[Room]
    teachers: list[Teacher]
    student_groups: list[StudentGroup]

    def __init__(self, timeslots, rooms, timeslotables, teachers, student_groups):
        self.timeslots = assign_ids(timeslots)
        self.rooms = assign_ids(rooms)
        self.timeslotables = assign_ids(timeslotables)
        self.lessons = assign_ids(get_lessons(timeslotables))
        self.teachers = assign_ids(teachers)
        self.student_groups = assign_ids(student_groups)

    def __str__(self):
        return (
            f"Timetable("
            f"timeslots={format_list(self.timeslots)},\n"
            f"rooms={format_list(self.rooms)},\n"
            f"timeslotables={format_list(self.timeslotables)},\n"
            f"score={str(self.score.toString()) if self.score is not None else 'None'}"
            f")"
        )

    def clear(self):
        for timeslotable in self.timeslotables:
            timeslotable.clear()

    
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


    def find_rooms_with_free_consecutive_timeslots_resursive(self, n_rooms, n_timeslots, rooms, timeslots, depth):
        for ri, room in enumerate(rooms):
            free_timeslots = self.find_free_timeslots_for_room(room)
            consecutive_timeslots = self.find_consecutive_timeslots(n_timeslots, free_timeslots)
            if consecutive_timeslots:
                self.find_rooms_with_free_consecutive_timeslots_resursive(n_rooms, n_timeslots, rooms[ri:], timeslots, depth + 1)

        pass

    def find_rooms_with_free_timeslots(self, timeslots):
        rooms = []
        for ri, room in enumerate(self.rooms):
            free_timeslots = self.find_free_timeslots_for_room(room)
            intersect = intersection(free_timeslots, timeslots)
            if len(intersect) == len(timeslots):
                rooms.append(room)

        return rooms


    # Find N rooms all that have T free timeslots in common
    def find_rooms_with_free_consecutive_timeslots(self, n_rooms, n_timeslots):
        for ri, room in enumerate(self.rooms):
            free_timeslots = self.find_free_timeslots_for_room(room)
            consecutive_timeslots = self.find_consecutive_timeslots(n_timeslots, free_timeslots)
            if consecutive_timeslots:
                rooms = self.find_rooms_with_free_timeslots(consecutive_timeslots)
                if len(rooms) >= n_rooms:
                    return rooms[:n_rooms], consecutive_timeslots

        print("Oh dear, need to improve code in find_rooms_with_free_consecutive_timeslots")
        return [], []

    def ordered_layout(self):
        """ Do a simple layout where each timeslotable is just placed down in order """
        self.clear()

        #self.lessons = get_lessons(self.timeslotables)

        for timeslotable in self.timeslotables:
            n_timeslots = timeslotable.n_timeslots
            found = False

            lessons = []
            if timeslotable.is_elective():
                lessons = timeslotable.lessons
            else:
                lessons = [timeslotable]

            n_lessons = len(lessons)
            rooms, timeslots = self.find_rooms_with_free_consecutive_timeslots(n_lessons, lessons[0].n_timeslots)
            for (lesson, room) in zip(lessons, rooms):
                lesson.set_timeslots(timeslots)
                lesson.set_room(room)


                #for ri, room in enumerate(self.rooms):
                #    free_timeslots = self.find_free_timeslots_for_room(room)
                #    consecutive_timeslots = self.find_consecutive_timeslots(n_timeslots, free_timeslots)
                #    if consecutive_timeslots:
                #        lesson.set_timeslots(consecutive_timeslots)
                #        lesson.set_room(room)
                #        found = True
                #        break

                #if not found:
                #    print("Not enough rooms to fit the lessons in. Try breaking down subjects into smaller lessons?")
                #    return
        
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

