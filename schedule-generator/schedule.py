# For now we assume 1 subject per day
# a subject is taught once per week
# Teacher always teaches the same subject

import json
import random

class Day:
    def __init__(self, data) -> None:
        self.data = data



class Subject:
    def __init__(self, data) -> None:
        self.data = data


class Teacher:
    def __init__(self, data) -> None:
        self.data = data

    def subject_teacher_pairs(self, subjects):
        return []



class Schedule:
    def __init__(self, data) -> None:
        self.days = []
        self.teachers = []
        self.subjects = []

    def load(self, filename):
        with open(filename, 'r') as f:
            dict = json.load(f)

            for day in dict['days']:
                self.days.append(Day(day))

            for teacher in dict['teachers']:
                self.teachers.append(Teacher(teacher))

            for subjects in dict['subjects']:
                self.subjects.append(Subject(subjects))

    def subject_teacher_pairs(self):
        """Generate subject-teacher pairs"""
        subject_teacher_pairs = []
        for teacher in self.teachers:
            subject_teacher_pairs = subject_teacher_pairs + teacher.subject_teacher_pairs(self.subjects)

        return subject_teacher_pairs


    def num_model_outputs(self):
        return len(self.days) * len(self.subjects) * len(self.teachers)

    def random_sample(self):
        """Generate a random schedule, that may or may not confirm to the rules"""
        s = [0] * self.num_model_outputs()
        #for day in self.days:
        #    for subject in self.subjects:

        return s


# model some 2d equation for testing
class SampleSchedule:
    def num_model_outputs(self):
        return 1

    def reward(self, data):
        return data[0] ** 4 - 20 * data[0] ** 2 +  10 * data[0] + 4 + data[0] * 20

    def random_sample(self):
        return [random.uniform(-5.0, 5.0)]


    def random_samples(self, n_samples):
        """Get N Nrandom samples ready for a training a model"""
        x = []
        y = []
        for _ in range(n_samples):
            x_r = self.random_sample()
            x.append(x_r)

            y_r = self.reward(x_r)
            y.append([y_r])

        return x, y

