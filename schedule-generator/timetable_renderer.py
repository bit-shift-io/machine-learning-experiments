from timetable import TimeTable
import pygame
import numpy as np

class TimetableRenderer:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode):
        self.window_size = 512  # The size of the PyGame window

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


    def render(self, timetable: TimeTable):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.font = pygame.font.SysFont(None, 14)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        timetable.print()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        room_list = timetable.room_list
        lesson_list = timetable.lesson_list
        timeslot_list = timetable.timeslot_list

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

        pix_square_size = (
            self.window_size / (len(room_list) + 1),
            self.window_size / (len(timeslot_list) + 1)
        )  # The size of a single grid square in pixels

        # Draw room labels
        for ri in range(len(room_list)):
            img = self.font.render(room_list[ri].name, False, 0)
            canvas.blit(img, (pix_square_size[0] * (ri + 1), 0))

        # Draw timetable labels
        for ti in range(len(timeslot_list)):
            timeslot = timeslot_list[ti]
            label = (timeslot.day_of_week[0:3] + " " + str(timeslot.start_time))[0:10]
            img = self.font.render(label, False, 0)
            canvas.blit(img, (0, pix_square_size[1] * (ti + 1)))


        # vertical lines
        for x in range(1, len(room_list) + 1):
            pygame.draw.line(
                    canvas,
                    0,
                    (pix_square_size[0] * x, 0),
                    (pix_square_size[0] * x, self.window_size),
                    width=1,
                )

        # horizontal lines
        for y in range(1, len(timeslot_list) + 1):
            pygame.draw.line(
                    canvas,
                    0,
                    (0, pix_square_size[1] * y),
                    (self.window_size, pix_square_size[1] * y),
                    width=1,
                )

        """
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
"""
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()