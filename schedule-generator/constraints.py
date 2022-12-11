from optapy import constraint_provider
from optapy.score import HardSoftScore
from optapy.constraint import ConstraintFactory, Joiners
from timetable import Lesson
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from timetable import TimeTable

# Trick since timedelta only works with datetime instances
today = date.today()


@dataclass
class Constraints:
    constraints: list

    def __init__(self, constraints):
        self.constraints = constraints

    def test(self, timetable: TimeTable):
        total_hard_score = 0
        total_soft_score = 0
        for constraint in self.constraints:
            hard, soft = constraint.test(self, timetable)
            total_hard_score += hard
            total_soft_score += soft

        return total_hard_score, total_soft_score

    def max_score(self, timetable: TimeTable):
        total_hard_score = 0
        total_soft_score = 0
        for constraint in self.constraints:
            hard, soft = constraint.max_score(self, timetable)
            total_hard_score += hard
            total_soft_score += soft

        return total_hard_score, total_soft_score

    

class Constraint:
    def test(c: Constraints, t: TimeTable):
        return 0, 0

    def max_score(c: Constraints, t: TimeTable):
        return 0, 0


    
class RoomConflict(Constraint):
    def test(self, c: Constraints, t: TimeTable):
        h = 0
        s = 0
        checked = []
        for l in t.get_lesson_list():
            r = list(filter(lambda l2: l2 not in checked and l != l2 and l2.timeslot == l.timeslot, t.get_lesson_list()))
            checked += r
            h -= len(r)

        return h, s

    def max_score(self, c: Constraints, t: TimeTable):
        return len(t.get_lesson_list()), 0


"""

def within_30_minutes(lesson1: Lesson, lesson2: Lesson):
    between = datetime.combine(today, lesson1.timeslot.end_time) - datetime.combine(today, lesson2.timeslot.start_time)
    return timedelta(minutes=0) <= between <= timedelta(minutes=30)


# Type annotation not needed, but allows you to get autocompletion
@constraint_provider
def define_constraints(constraint_factory: ConstraintFactory):
    return [
        # Hard constraints
        room_conflict(constraint_factory),
        teacher_conflict(constraint_factory),
        student_group_conflict(constraint_factory),
        # Soft constraints
        teacher_room_stability(constraint_factory),
        teacher_time_efficiency(constraint_factory),
        student_group_subject_variety(constraint_factory)
    ]


def room_conflict(constraint_factory: ConstraintFactory):
    # A room can accommodate at most one lesson at the same time.
    return constraint_factory \
        .for_each(Lesson) \
        .join(Lesson,
              # ... in the same timeslot ...
              Joiners.equal(lambda lesson: lesson.timeslot),
              # ... in the same room ...
              Joiners.equal(lambda lesson: lesson.room),
              # form unique pairs
              Joiners.less_than(lambda lesson: lesson.id)
              ) \
        .penalize("Room conflict", HardSoftScore.ONE_HARD)


def teacher_conflict(constraint_factory: ConstraintFactory):
    # A teacher can teach at most one lesson at the same time.
    return constraint_factory \
        .for_each(Lesson) \
        .join(Lesson,
              Joiners.equal(lambda lesson: lesson.timeslot),
              Joiners.equal(lambda lesson: lesson.teacher),
              Joiners.less_than(lambda lesson: lesson.id)
              ) \
        .penalize("Teacher conflict", HardSoftScore.ONE_HARD)


def student_group_conflict(constraint_factory: ConstraintFactory):
    # A student can attend at most one lesson at the same time.
    return constraint_factory \
        .for_each(Lesson) \
        .join(Lesson,
              Joiners.equal(lambda lesson: lesson.timeslot),
              Joiners.equal(lambda lesson: lesson.student_group),
              Joiners.less_than(lambda lesson: lesson.id)
              ) \
        .penalize("Student group conflict", HardSoftScore.ONE_HARD)


def teacher_room_stability(constraint_factory: ConstraintFactory):
    # A teacher prefers to teach in a single room.
    return constraint_factory \
        .for_each(Lesson) \
        .join(Lesson,
              Joiners.equal(lambda lesson: lesson.teacher),
              Joiners.less_than(lambda lesson: lesson.id)
              ) \
        .filter(lambda lesson1, lesson2: lesson1.room != lesson2.room) \
        .penalize("Teacher room stability", HardSoftScore.ONE_SOFT)


def teacher_time_efficiency(constraint_factory: ConstraintFactory):
    # A teacher prefers to teach sequential lessons and dislikes gaps between lessons.
    return constraint_factory \
        .for_each(Lesson) \
        .join(Lesson,
              Joiners.equal(lambda lesson: lesson.teacher),
              Joiners.equal(lambda lesson: lesson.timeslot.day_of_week)
              ) \
        .filter(within_30_minutes) \
        .reward("Teacher time efficiency", HardSoftScore.ONE_SOFT)


def student_group_subject_variety(constraint_factory: ConstraintFactory):
    # A student group dislikes sequential lessons on the same subject.
    return constraint_factory \
        .for_each(Lesson) \
        .join(Lesson,
              Joiners.equal(lambda lesson: lesson.subject),
              Joiners.equal(lambda lesson: lesson.student_group),
              Joiners.equal(lambda lesson: lesson.timeslot.day_of_week)
              ) \
        .filter(within_30_minutes) \
        .penalize("Student group subject variety", HardSoftScore.ONE_SOFT)

"""