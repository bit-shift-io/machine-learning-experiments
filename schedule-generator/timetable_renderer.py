from timetable import Timetable
from constraints import Constraints
import pygame
import numpy as np

RED = (100, 0, 0)
GREEN = (0, 100, 0)
BLUE = (0, 0, 100)


line_height = 10

class TimetableRenderer:
    metadata = {"render_modes": ["human", "human_fast", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode):
        self.window_size = (1500, 900)  # The size of the PyGame window

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def render(self, timetable: Timetable, constraints: Constraints):
        if self.window is None and (self.render_mode == "human" or self.render_mode == "human_fast"):
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.font = pygame.font.SysFont(None, 16)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        #timetable.print()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        rooms = timetable.rooms
        lessons = timetable.lessons
        timeslots = timetable.timeslots

        header_size = (40, 10)

        pix_square_size = (
            (self.window_size[0] - header_size[0]) / len(rooms),
            (self.window_size[1] - header_size[1]) / len(timeslots)
        )  # The size of a single grid square in pixels

        pygame.draw.line(
                    canvas,
                    0,
                    (0, header_size[1]),
                    (self.window_size[0], header_size[1]),
                    width=1,
                )

        pygame.draw.line(
                    canvas,
                    0,
                    (header_size[0], 0),
                    (header_size[0], self.window_size[1]),
                    width=1,
                )

        
        # Draw room labels
        for ri in range(len(rooms)):
            img = self.font.render(rooms[ri].name, True, 0)
            canvas.blit(img, (header_size[0] + (pix_square_size[0] * ri), 0))

        # Draw timetable labels
        for ti in range(len(timeslots)):
            timeslot = timeslots[ti]
            img = self.font.render(timeslot.day_of_week[0:2], True, 0)
            canvas.blit(img, (0, header_size[1] + (pix_square_size[1] * ti)))

            label = (str(timeslot.start_time))[0:5]
            img = self.font.render(label, True, 0)
            canvas.blit(img, (0, header_size[1] + (pix_square_size[1] * ti) + line_height))



        # vertical lines
        for x in range(len(rooms)):
            pygame.draw.line(
                    canvas,
                    0,
                    (header_size[0] + (pix_square_size[0] * x), 0),
                    (header_size[0] + (pix_square_size[0] * x), self.window_size[1]),
                    width=1,
                )
        
        # horizontal lines
        for y in range(len(timeslots)):
            pygame.draw.line(
                    canvas,
                    0,
                    (0, header_size[1] + (pix_square_size[1] * y)),
                    (self.window_size[0], header_size[1] + (pix_square_size[1] * y)),
                    width=1,
                )

        # draw contents of each cell
        for ri in range(len(rooms)):
            for ti in range(len(timeslots)):
                start_pos = (header_size[0] + (pix_square_size[0] * ri), 
                            header_size[1] + (pix_square_size[1] * ti))
                room = rooms[ri]
                timeslot = timeslots[ti]
                self.draw_room_timeslot(timetable, constraints, canvas, room, timeslot, start_pos, pix_square_size)


        # draw score
        max_hard_score, max_soft_score = constraints.max_score(timetable)
        hard_score, soft_score = constraints.test(timetable)
        label = str(hard_score) + " / " + str(max_hard_score)
        img = self.font.render(label, True, GREEN if hard_score == max_hard_score else RED)
        canvas.blit(img, (0, 0))


        if self.render_mode == "human" or self.render_mode == "human_fast":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            if self.clock != None:
                self.clock.tick(self.metadata["render_fps"])

        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def draw_room_timeslot(self, timetable, constraints, canvas, room, timeslot, start_pos, size):
        lessons = timetable.lessons
        lessons = list(filter(lambda l: l.room == room and timeslot in l.timeslots, lessons))


        #sub_timetable = Timetable([timeslot], [room], lessons, [], [])
        #max_hard_score, max_soft_score = constraints.max_score(sub_timetable)
        #hard_score, soft_score = constraints.test(sub_timetable)

        for li, lesson in enumerate(lessons):
            label = lesson.subject
            img = self.font.render(label, True, 0)
            canvas.blit(img, (start_pos[0], start_pos[1] + (line_height * li)))

            label = lesson.teacher.name
            img = self.font.render(label, True, 0)
            canvas.blit(img, (start_pos[0], start_pos[1] + (line_height * (li + 1))))

        # draw the score for this cell
        #label = str(hard_score)# + " | " + str(max_hard_score)
        #img = self.font.render(label, True, GREEN if hard_score == max_hard_score else RED)
        #canvas.blit(img, (start_pos[0], start_pos[1] + size[1] - line_height))


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()