import datetime
import time


class ProgressInformer:
    """
    Object that enables showing the progress of some operation in the console.
    """

    def __init__(self, caption='', length=20):
        """
        Creates a new ProgressInformer instance.

        :param caption: Text to the left of the progressbar
        :param length: Length of the progressbar in characters
        """
        self.progress = 0
        self.start_time = int(time.time())
        self.time_elapsed = 0
        self.length = length
        self.caption = caption

    def report_progress(self, progress: float):
        """
        Updates progress data stored in the object.
        The progressbar is updated 5 times a second.
        """
        current_elapsed = int(time.time()) - self.start_time
        if current_elapsed - self.time_elapsed > 0.2:
            current_pct = int(100 * progress)
            bar_count = int(current_pct * self.length / 100)
            d_seconds = current_elapsed - self.time_elapsed
            d_progress = progress - self.progress
            estimated_left = (1 - progress) * d_seconds / d_progress
            print('\r{} [{}] {}% ETA: {}'.format(
                self.caption,
                '#' * bar_count + '-' * (self.length - bar_count),
                current_pct,
                str(datetime.timedelta(seconds=int(estimated_left)))),
                end='')
            self.time_elapsed = current_elapsed
            self.progress = progress

    def finish(self):
        """
        Prints a full progress bar and goes to next line
        """
        print('\r{} [{}] 100% ETA: 0:00:00'.format(self.caption, '#' * self.length))
