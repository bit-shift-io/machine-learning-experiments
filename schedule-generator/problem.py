from timetable import Timetable, Timeslot, Room, Lesson, Teacher, StudentGroup
from datetime import time
from constraints import Constraints, RoomConflict, TeacherConflict, StudentGroupConflict

def constraint_list():
    constraints = [
        RoomConflict(0),
        StudentGroupConflict(1), # this is killing the learning.... how can we improve it? we should do an initial layout based on student group. maybe AI actions are swap actions instead of move?!
        TeacherConflict(2),
    ]
    return Constraints(constraints)

def day_timeslots_7(day, start_idx):
    return [
        Timeslot(start_idx+0, day, time(hour=8, minute=0), time(hour=9, minute=0)),
        Timeslot(start_idx+1, day, time(hour=9, minute=0), time(hour=10, minute=0)),
        Timeslot(start_idx+2, day, time(hour=10, minute=0), time(hour=11, minute=0)),
        Timeslot(start_idx+3, day, time(hour=11, minute=0), time(hour=12, minute=0)),
        Timeslot(start_idx+4, day, time(hour=12, minute=0), time(hour=14, minute=0)),
        Timeslot(start_idx+5, day, time(hour=13, minute=0), time(hour=15, minute=0)),
        Timeslot(start_idx+6, day, time(hour=14, minute=0), time(hour=16, minute=0)),
    ]

def generate_problem_large():
    teacher_list = [Teacher(i, f"Teacher {i}") for i in range(40)]
    student_group_list = [StudentGroup(i, f"Grade {i+8}") for i in range(5)]
    timeslot_list = [] + day_timeslots_7('MON', 0) + day_timeslots_7('TUES', 7) + day_timeslots_7('WED', 14) + day_timeslots_7('THURS', 21) + day_timeslots_7('FRI', 28)
    room_list = [StudentGroup(i, f"Room {i}") for i in range(33)]

    # each grade (student group) needs 15 subjects
    subjects = ['Math 1', 'Math 2', 'Physics', 'English', 'Chemistry', 'Biology', 'Geography', 'Economics', 'Tech Studies', 'Gym', 'Home Ec', 'Religion', 'Phsycology', 'I.T.', 'French']
    lesson_list = []
    tii = 0
    ti = 0
    ti_inc = len(teacher_list) / len(subjects) # how many subjects does each teacher need to teach to cover every lesson?
    for sgi, sg in enumerate(student_group_list):
        for si in range(len(subjects)):
            lesson_list.append(Lesson(0, f"{subjects[si]} for Grade {si}", teacher_list[ti], student_group_list[sgi]))
            tii += 1
            if tii >= ti_inc:
                tii = 0
                ti += 1

    return Timetable(timeslot_list, room_list, lesson_list, teacher_list, student_group_list)


def generate_problem_medium():
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



def generate_problem_small():
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
    ]
    room_list = [
        Room(0, "Room A"),
        Room(1, "Room B"),
        Room(2, "Room C")
    ]
    lesson_list = [
        Lesson(0, "Math", teacher_list[0], student_group_list[0]),
        Lesson(1, "Math", teacher_list[0], student_group_list[0]),
        Lesson(2, "Physics", teacher_list[1], student_group_list[0]),
        Lesson(3, "Chemistry",teacher_list[1], student_group_list[0]),
        Lesson(4, "Biology", teacher_list[2], student_group_list[0]),
        Lesson(5, "History", teacher_list[3], student_group_list[0]),
        Lesson(6, "English", teacher_list[3], student_group_list[0]),
        Lesson(7, "English", teacher_list[3], student_group_list[0]),
        Lesson(8, "Spanish", teacher_list[4], student_group_list[0]),
        Lesson(9, "Spanish", teacher_list[4], student_group_list[0]),
    ]
    return Timetable(timeslot_list, room_list, lesson_list, teacher_list, student_group_list)
