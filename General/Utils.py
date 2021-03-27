import datetime
import time


class ProgressInformer:
    """
    Object that enables showing the progress of some operation in the console.
    """

    def __init__(self, **kwargs):
        """
        Creates a new ProgressInformer instance.

        :param caption: Text to the left of the progressbar
        :param length: Length of the progressbar in characters
        """
        if 'verbose' in kwargs and kwargs['verbose'] == False:
            self.snapshots = None
            return
        self.snapshots = [[int(time.time()), 0.]]
        self.length = kwargs.get('length', 20)
        self.caption = kwargs.get('caption', '')
        self.max = kwargs.get('max', 1)

    def report_progress(self, progress):
        """
        Updates progress data stored in the object.
        The progressbar is updated 5 times a second.
        """
        if self.snapshots is None:
            return
        progress /= self.max
        current_dt = int(time.time()) - self.snapshots[-1][0]
        if current_dt > 0.2 and progress > self.snapshots[-1][1]:
            self.snapshots.append([int(time.time()), progress])
            current_pct = int(100 * progress)
            bar_count = int(current_pct * self.length / 100)
            d_seconds = self.snapshots[-1][0] - self.snapshots[0][0]
            d_progress = self.snapshots[-1][1] - self.snapshots[0][1]
            estimated_left = (1 - progress) * d_seconds / d_progress
            print('\r{} [{}] {}% ETA: {}'.format(
                self.caption,
                '#' * bar_count + '-' * (self.length - bar_count),
                current_pct,
                str(datetime.timedelta(seconds=int(estimated_left)))),
                end='')
            self.snapshots = self.snapshots[-5:]
        else:
            self.snapshots[-1][1] = progress

    def report_increment(self):
        if self.snapshots is None:
            return
        self.report_progress(self.snapshots[-1][1] * self.max + 1)

    def finish(self):
        """
        Prints a full progress bar and goes to next line
        """
        if self.snapshots is not None:
            print('\r{} [{}] 100% ETA: 0:00:00'.format(self.caption, '#' * self.length))
