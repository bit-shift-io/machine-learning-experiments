from timetable import Timetable, Timeslot, Room, Lesson, Teacher, StudentGroup
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
    teacher_list = [
        Teacher(0, "A. Turing"),
        Teacher(1, "M. Curie"),
        Teacher(2, "C. Darwin"),
        Teacher(3, "I. Jones"),
        Teacher(4, "P. Cruz")
    ]

    student_group_list = [
        StudentGroup(0, "9th grade"),
        StudentGroup(1, "10th grade")
    ]

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

    # subject is just a string unless the model needs to apply some subject constraints
    lesson_list = [
        Lesson(0, "Math", teacher_list[0], student_group_list[0]),
        Lesson(1, "Math", teacher_list[0], student_group_list[0]),
        Lesson(2, "Physics", teacher_list[1], student_group_list[0]),
        Lesson(3, "Chemistry", teacher_list[1], student_group_list[0]),
        Lesson(4, "Biology", teacher_list[2], student_group_list[0]),
        Lesson(5, "History", teacher_list[3], student_group_list[0]),
        Lesson(6, "English", teacher_list[3], student_group_list[0]),
        Lesson(7, "English", teacher_list[3], student_group_list[0]),
        Lesson(8, "Spanish", teacher_list[4], student_group_list[0]),
        Lesson(9, "Spanish", teacher_list[4], student_group_list[0]),
        Lesson(10, "Math", teacher_list[0], student_group_list[1]),
        Lesson(11, "Math", teacher_list[0], student_group_list[1]),
        Lesson(12, "Math", teacher_list[0], student_group_list[1]),
        Lesson(13, "Physics", teacher_list[1], student_group_list[1]),
        Lesson(14, "Chemistry", teacher_list[1], student_group_list[1]),
        Lesson(15, "French", teacher_list[1], student_group_list[1]),
        Lesson(16, "Geography", teacher_list[2], student_group_list[1]),
        Lesson(17, "History", teacher_list[3], student_group_list[1]),
        Lesson(18, "English", teacher_list[4], student_group_list[1]),
        Lesson(19, "Spanish", teacher_list[4], student_group_list[1]),
    ]

    return Timetable(timeslot_list, room_list, lesson_list, teacher_list, student_group_list)


"""
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
"""