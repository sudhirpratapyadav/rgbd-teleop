import time
from collections import deque
from tabulate import tabulate
import numpy as np

class TimingLogger:
    def __init__(self, max_iters=1000):
        # Stores timing data for each step
        self.step_names = []
        self.timing_data = {}
        self.max_iters = max_iters
        self.base_times = deque(maxlen=self.max_iters)
    
    def next(self):
        """ Start the timer. """
        self.base_times.append(time.time())
    
    def stamp(self, step_name):
        """ Log the time for a step. """
        current_time = time.time()
        if step_name not in self.step_names:
            self.step_names.append(step_name)
            self.timing_data[step_name] = deque(maxlen=self.max_iters)
        self.timing_data[step_name].append(current_time)
    
    def report(self, include_iters=True, time_unit='s', align='center', precision=None, ignore_init_iters=0):
        print('\n--------- Timing report start ---------')

        elapsed_times = {}
        for step_name in self.step_names:
            elapsed_times[step_name] = []
        
        num_iters = len(self.base_times)

        for i in range(num_iters):
            for step_num, step_name in enumerate(self.step_names):
                if step_num == 0:
                    prev_time = self.base_times[i]
                else:
                    prev_time = self.timing_data[self.step_names[step_num - 1]][i]
                elapsed_time = self.timing_data[step_name][i] - prev_time
                if time_unit == 'ms':
                    elapsed_time *= 1000
                elapsed_times[step_name].append(elapsed_time)
        
        if precision is None:
            precision = 3 if time_unit == 's' else 0
        
        headers = ["", "Total", "Mean", "Max", "Min", "Num"]
        table_data = []
        for step in self.step_names:
            times = np.array(elapsed_times[step])[ignore_init_iters:]
            row = [
                step,
                np.sum(times),
                np.mean(times),
                np.max(times),
                np.min(times),
                len(times)
            ]
            table_data.append(row)

        print("\nSummary Statistics:")
        print(tabulate(table_data, headers=headers, floatfmt=f".{precision}f", numalign=align))

        if include_iters:
            print("\nIteration-wise breakdown:")
            headers = ["Iter."] + self.step_names
            table_data = []
            for i in range(ignore_init_iters, num_iters):
                row = [i+1] + [elapsed_times[step][i] for step in self.step_names]
                table_data.append(row)
            print(tabulate(table_data, headers=headers, floatfmt=f".{precision}f", numalign=align))

        print('--------- Timing report end ---------\n')

# Usage
if __name__ == '__main__':
    logger = TimingLogger()
    for i in range(10):
        logger.next()
        time.sleep(0.1)
        logger.stamp('read_frame')
        time.sleep(0.1)
        logger.stamp('pose_detect')
        time.sleep(0.1)
        logger.stamp('keypoints')
        time.sleep(0.1)
        logger.stamp('display')
    logger.report(time_unit='ms')