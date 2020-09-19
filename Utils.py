import datetime
import time
import colorsys


class ProgressInformer:
    def __init__(self, title):
        self.progress_pct = 0
        self.start_time = int(time.time())
        self.time_elapsed = 0
        self.title = title

    def report_progress(self, progress):
        current_pct = int(100 * progress)
        current_elapsed = int(time.time()) - self.start_time
        if current_pct != self.progress_pct or current_elapsed != self.time_elapsed:
            print('\r{} [{:-<20}] {}% ETA: {}'.format(
                self.title, '#' * int(current_pct / 5),
                current_pct,
                str(datetime.timedelta(seconds=int(self.time_elapsed * (1 / progress - 1))))),
                end='')
            self.time_elapsed = current_elapsed
            self.progress_pct = current_pct

    def finish(self):
        print('\r{} [{}] 100% ETA: 0:00:00'.format(self.title, '#' * 20))


def color_by_hue(hue):
    return colorsys.hsv_to_rgb(hue, 1, 1)
