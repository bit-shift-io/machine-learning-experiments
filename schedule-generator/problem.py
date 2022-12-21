from timetable import Timetable, Timeslot, Room, Lesson, Teacher, StudentGroup, Elective, assign_ids
from datetime import time
from constraints import Constraints, RoomConflict, TeacherConflict, StudentGroupConflict
import math

def constraint_list():
    constraints = [
        RoomConflict(0),
        StudentGroupConflict(1), # this is killing the learning.... how can we improve it? we should do an initial layout based on student group. maybe AI actions are swap actions instead of move?!
        TeacherConflict(2),
    ]
    return Constraints(constraints)


def day_timeslots_7(day):
    return [
        Timeslot(day, time(hour=8, minute=0), time(hour=9, minute=0)),
        Timeslot(day, time(hour=9, minute=0), time(hour=10, minute=0)),
        Timeslot(day, time(hour=10, minute=0), time(hour=11, minute=0)),
        Timeslot(day, time(hour=11, minute=0), time(hour=12, minute=0)),
        Timeslot(day, time(hour=12, minute=0), time(hour=13, minute=0)),
        Timeslot(day, time(hour=13, minute=0), time(hour=14, minute=0)),
        Timeslot(day, time(hour=14, minute=0), time(hour=15, minute=0)),
    ]

# 5 hours of lessons in 3 sessions
def subject_lessons_5(subject_name, teacher, student_group):
    return [
        Lesson(f"G{student_group.id+7} {subject_name} (1/3) DBL", teacher, student_group, 2),
        Lesson(f"G{student_group.id+7} {subject_name} (2/3) DBL", teacher, student_group, 2),
        Lesson(f"G{student_group.id+7} {subject_name} (3/3) SGL", teacher, student_group, 1)
    ]

# 3 hours worth of lessons
def subject_lessons_3(subject_name, teacher, student_group):
    return [
        Lesson(f"G{student_group.id+7} {subject_name} (1/2) DBL", teacher, student_group, 2),
        Lesson(f"G{student_group.id+7} {subject_name} (2/2) SGL", teacher, student_group, 1)
    ]

# 2 hours worth of lessons
def subject_lessons_2(subject_name, teacher, student_group):
    return [
        Lesson(f"G{student_group.id+7} {subject_name} (1/1) DBL", teacher, student_group, 2)
    ]


def generate_problem_large():
    n_teachers = 40
    n_rooms = 6 #33 # cut down so we can view all the rooms
    n_student_groups = 6

    teachers = [Teacher(f"Teacher {i}") for i in range(n_teachers)]
    student_groups = assign_ids([StudentGroup(f"Grade {i+7}") for i in range(n_student_groups)])
    timeslots = [] + day_timeslots_7('MON') + day_timeslots_7('TUES') + day_timeslots_7('WED') + day_timeslots_7('THURS') + day_timeslots_7('FRI')
    rooms = [Room(f"Room {i}") for i in range(n_rooms)]

    # only 7 subjects fit into 35 lessons per week with 5 each subject taking 5 lessons! each grade (student group) needs 15 subjects
    subjects = ['Math', 'Geography', 'Physics', 'English', 'Elective A', 'Elective B', 'Elective C']
    #subjects = ['Math 1', 'Math 2', 'Physics', 'English', 'Chemistry', 'Biology', 'Geography', 'Economics', 'Tech Studies', 'Gym', 'Home Ec', 'Religion', 'Phsycology', 'I.T.', 'French']
    lessons = [] # Lessons + Electives (anything Timetableable)
    ti = 0


    # Year 7: 5 lessons each for English, Maths, Science, Humanities, 3 lessons each for German, PE, 2 for Christian living. That is 28 lessons so far. 2 for an arts elective line (music or drama, the class is split in to and will do one semester of each, 
    # but essentially requires 2 class rooms with 2 teachers for the year), then there is another elective line which has 5 lessons. This is split between design tech, home ec, art and digital tech. They do digital tech / art for a term each, then home ec 
    # and tech for term each. The digital tech/art and home ec/design tech semesters swap over with year 8 (as year 7 digital tech/art happens, year 8s do home ec/design tech, then they switch over). So you would need to be running this elective at the same 
    # time as a year 8 class. (there are 3 classes per year level).
    lessons += subject_lessons_5('English', teachers[0], student_groups[0])
    lessons += subject_lessons_5('Maths', teachers[0], student_groups[0])
    lessons += subject_lessons_5('Science', teachers[0], student_groups[0])
    lessons += subject_lessons_5('Humanities', teachers[0], student_groups[0])

    lessons += subject_lessons_3('German', teachers[0], student_groups[0])
    lessons += subject_lessons_3('PE', teachers[0], student_groups[0])

    lessons += subject_lessons_2('Christian Living', teachers[0], student_groups[0])

    # art elective
    drama = subject_lessons_2('Drama - Art E.', teachers[0], student_groups[0])
    music = subject_lessons_2('Music - Art E.', teachers[0], student_groups[0])
    lessons += Elective.zipLessons([drama, music], "G7 Elective A")

    # another elective
    design_tech = subject_lessons_5("Design Tech / Art", teachers[0], student_groups[0])
    home_ec = subject_lessons_5("Home Ec / Tech", teachers[0], student_groups[0])
    lessons += Elective.zipLessons([design_tech, home_ec], "G7 Elective B")

    """
    for sgi, sg in enumerate(student_groups):
        for si in range(len(subjects)):
            subject_lessons = subject_lessons_5(len(lessons), subjects[si], teachers[ti], student_groups[sgi])
            lessons += subject_lessons
            ti += 1
            ti = ti % len(teachers)
    """

    return Timetable(assign_ids(timeslots), assign_ids(rooms), assign_ids(lessons), assign_ids(teachers), student_groups)


def generate_problem_medium():
    teachers = [
        Teacher(0, "A. Turing"),
        Teacher(1, "M. Curie"),
        Teacher(2, "C. Darwin"),
        Teacher(3, "I. Jones"),
        Teacher(4, "P. Cruz")
    ]
    student_groups = [
        StudentGroup(0, "9th grade"),
        StudentGroup(1, "10th grade")
    ]
    timeslots = [
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
    rooms = [
        Room(0, "Room A"),
        Room(1, "Room B"),
        Room(2, "Room C")
    ]
    # subject is just a string unless the model needs to apply some subject constraints
    lessons = [
        Lesson(0, "Math", teachers[0], student_groups[0]),
        Lesson(1, "Math", teachers[0], student_groups[0]),
        Lesson(2, "Physics", teachers[1], student_groups[0]),
        Lesson(3, "Chemistry", teachers[1], student_groups[0]),
        Lesson(4, "Biology", teachers[2], student_groups[0]),
        Lesson(5, "History", teachers[3], student_groups[0]),
        Lesson(6, "English", teachers[3], student_groups[0]),
        Lesson(7, "English", teachers[3], student_groups[0]),
        Lesson(8, "Spanish", teachers[4], student_groups[0]),
        Lesson(9, "Spanish", teachers[4], student_groups[0]),
        Lesson(10, "Math", teachers[0], student_groups[1]),
        Lesson(11, "Math", teachers[0], student_groups[1]),
        Lesson(12, "Math", teachers[0], student_groups[1]),
        Lesson(13, "Physics", teachers[1], student_groups[1]),
        Lesson(14, "Chemistry", teachers[1], student_groups[1]),
        Lesson(15, "French", teachers[1], student_groups[1]),
        Lesson(16, "Geography", teachers[2], student_groups[1]),
        Lesson(17, "History", teachers[3], student_groups[1]),
        Lesson(18, "English", teachers[4], student_groups[1]),
        Lesson(19, "Spanish", teachers[4], student_groups[1]),
    ]
    return Timetable(timeslots, rooms, lessons, teachers, student_groups)



def generate_problem_small():
    teachers = [
        Teacher(0, "A. Turing"),
        Teacher(1, "M. Curie"),
        Teacher(2, "C. Darwin"),
        Teacher(3, "I. Jones"),
        Teacher(4, "P. Cruz")
    ]
    student_groups = [
        StudentGroup(0, "9th grade"),
        StudentGroup(1, "10th grade")
    ]
    timeslots = [
        Timeslot(0, "MONDAY", time(hour=8, minute=30), time(hour=9, minute=30)),
        Timeslot(1, "MONDAY", time(hour=9, minute=30), time(hour=10, minute=30)),
        Timeslot(2, "MONDAY", time(hour=10, minute=30), time(hour=11, minute=30)),
        Timeslot(3, "MONDAY", time(hour=13, minute=30), time(hour=14, minute=30)),
        Timeslot(4, "MONDAY", time(hour=14, minute=30), time(hour=15, minute=30)),
    ]
    rooms = [
        Room(0, "Room A"),
        Room(1, "Room B"),
        Room(2, "Room C")
    ]
    lessons = [
        Lesson(0, "Math", teachers[0], student_groups[0]),
        Lesson(1, "Math", teachers[0], student_groups[0]),
        Lesson(2, "Physics", teachers[1], student_groups[0]),
        Lesson(3, "Chemistry",teachers[1], student_groups[0]),
        Lesson(4, "Biology", teachers[2], student_groups[0]),
        Lesson(5, "History", teachers[3], student_groups[0]),
        Lesson(6, "English", teachers[3], student_groups[0]),
        Lesson(7, "English", teachers[3], student_groups[0]),
        Lesson(8, "Spanish", teachers[4], student_groups[0]),
        Lesson(9, "Spanish", teachers[4], student_groups[0]),
    ]
    return Timetable(timeslots, rooms, lessons, teachers, student_groups)
