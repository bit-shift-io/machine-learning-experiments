from timetable import Timetable, Timeslot, Room, Lesson
from datetime import time
from constraints import Constraints, RoomConflict, TeacherConflict, StudentGroupConflict

def constraint_list():
    constraints = [
        RoomConflict(0),
        TeacherConflict(1),
        StudentGroupConflict(2)
    ]
    return Constraints(constraints)


def generate_problem():
    timeslot_list = [
        Timeslot(0, "MONDAY", time(hour=8, minute=30), time(hour=9, minute=30)),
        Timeslot(1, "MONDAY", time(hour=9, minute=30), time(hour=10, minute=30)),
        Timeslot(2, "MONDAY", time(hour=10, minute=30), time(hour=11, minute=30)),
        Timeslot(3, "MONDAY", time(hour=13, minute=30), time(hour=14, minute=30)),
        Timeslot(4, "MONDAY", time(hour=14, minute=30), time(hour=15, minute=30)),
        Timeslot(5, "TUESDAY", time(hour=8, minute=30), time(hour=9, minute=30)),
        Timeslot(6, "TUESDAY", time(hour=9, minute=30), time(hour=10, minute=30)),
        Timeslot(7, "TUESDAY", time(hour=10, minute=30), time(hour=11, minute=30)),
        Timeslot(8, "TUESDAY", time(hour=13, minute=30), time(hour=14, minute=30)),
        Timeslot(9, "TUESDAY", time(hour=14, minute=30), time(hour=15, minute=30)),
    ]
    room_list = [
        Room(0, "Room A"),
        Room(1, "Room B"),
        Room(2, "Room C")
    ]
    lesson_list = [
        Lesson(0, "Math", "A. Turing", "9th grade"),
        Lesson(1, "Math", "A. Turing", "9th grade"),
        Lesson(2, "Physics", "M. Curie", "9th grade"),
        Lesson(3, "Chemistry", "M. Curie", "9th grade"),
        Lesson(4, "Biology", "C. Darwin", "9th grade"),
        Lesson(5, "History", "I. Jones", "9th grade"),
        Lesson(6, "English", "I. Jones", "9th grade"),
        Lesson(7, "English", "I. Jones", "9th grade"),
        Lesson(8, "Spanish", "P. Cruz", "9th grade"),
        Lesson(9, "Spanish", "P. Cruz", "9th grade"),
        Lesson(10, "Math", "A. Turing", "10th grade"),
        Lesson(11, "Math", "A. Turing", "10th grade"),
        Lesson(12, "Math", "A. Turing", "10th grade"),
        Lesson(13, "Physics", "M. Curie", "10th grade"),
        Lesson(14, "Chemistry", "M. Curie", "10th grade"),
        Lesson(15, "French", "M. Curie", "10th grade"),
        Lesson(16, "Geography", "C. Darwin", "10th grade"),
        Lesson(17, "History", "I. Jones", "10th grade"),
        Lesson(18, "English", "P. Cruz", "10th grade"),
        Lesson(19, "Spanish", "P. Cruz", "10th grade"),
    ]

    return Timetable(timeslot_list, room_list, lesson_list)



def generate_problem_simple():
    timeslot_list = [
        Timeslot(0, "MONDAY", time(hour=8, minute=30), time(hour=9, minute=30)),
        Timeslot(1, "MONDAY", time(hour=9, minute=30), time(hour=10, minute=30)),
        Timeslot(2, "MONDAY", time(hour=10, minute=30), time(hour=11, minute=30)),
        Timeslot(3, "MONDAY", time(hour=13, minute=30), time(hour=14, minute=30)),
        Timeslot(4, "MONDAY", time(hour=14, minute=30), time(hour=15, minute=30)),
    ]
    room_list = [
        Room(0, "Room A"),
        Room(1, "Room B"),
        Room(2, "Room C")
    ]
    lesson_list = [
        Lesson(0, "Math", "A. Turing", "9th grade"),
        Lesson(1, "Math", "A. Turing", "9th grade"),
        Lesson(2, "Physics", "M. Curie", "9th grade"),
        Lesson(3, "Chemistry", "M. Curie", "9th grade"),
        Lesson(4, "Biology", "C. Darwin", "9th grade"),
        Lesson(5, "History", "I. Jones", "9th grade"),
        Lesson(6, "English", "I. Jones", "9th grade"),
        Lesson(7, "English", "I. Jones", "9th grade"),
        Lesson(8, "Spanish", "P. Cruz", "9th grade"),
        Lesson(9, "Spanish", "P. Cruz", "9th grade"),
    ]

    return Timetable(timeslot_list, room_list, lesson_list)
