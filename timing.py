import time

from typing import Sequence

class Timer:
    start_time: time
    runtime: time
    checkpoints: Sequence = []

    def __init__(self, max_time = None):
        self.max_time = max_time

    def start(self):
        self.start_time = time.perf_counter()
        self.runtime = 0

    def update(self):
        self.runtime = time.perf_counter() - self.start_time

    def out_of_time(self):
        if self.max_time is not None:
            return self.runtime >= self.max_time
        else:
            return False
        
    def create_checkpoint(self):
        self.update()
        self.checkpoints.append(self.runtime)

    def measure_checkpoints(self):
        measured_cp = []

        for i in range(1, len(self.checkpoints)):
            checkpoint_timing = self.checkpoints[i] - self.checkpoints[i - 1]
            measured_cp.append(checkpoint_timing)

        return measured_cp

    def print(self):
        self.update()
        print('total runtime: ', self.runtime)
        print('checkpoints: ', self.measure_checkpoints())