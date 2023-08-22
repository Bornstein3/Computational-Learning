import shutil
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils import _find_continuous_times_when_biger_then_threshold_for_record, plot_and_save, avg

data_file_path = "ND7_post_g0_t0.imec0.ap.SHORT.bin"


class Recordings:
    parsed_recordings: np.ndarray
    reshaped_recordings: np.ndarray
    avg_recordings: np.ndarray = None
    clusters: np.ndarray
    centroids: np.ndarray

    def main(self):
        print("hello")
        self.parse_recordings()
        self.plot_trajectory_for_electrode_at_time(electrode=50, time=(0, 3000000)) # first 100 seconds on the 50th electrode
        self.correlation_with_window()
        self.var_with_window()

    def parse_recordings(self):
        with open(file=data_file_path, mode='r') as file:
            parsed_recordings = np.fromfile(file=file, dtype=np.int16)
            parsed_recordings = parsed_recordings.reshape(int(parsed_recordings.shape[0] / 385), 385)
        self.parsed_recordings = parsed_recordings[:, :-1]
        self.reshape_recordings()

    def show_table(self, data=None):  # to print the reshaped recordings table use: self.show_table()
        data = data if data else self.reshaped_recordings
        df = pd.DataFrame(data)
        print(df)

    def reshape_recordings(self, time_interval=(0, 10800000), electrode_range=(0, 385)):
        self.reshaped_recordings = self.parsed_recordings[int(time_interval[0] * 30000): int(time_interval[1] * 30000),
                                   int(electrode_range[0]): int(electrode_range[1])]

    def correlation_with_window(self, window=0.1, should_zip=False):
        dir_name = 'output/corrcoef on single electrode with all electrodes with window'
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)

        for time in range(60, 360, 60):
            interval = (time, time + window)
            path = f'{dir_name}/corrcoef on single electrode with all electrodes with window at time {interval}'
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            for i in range(0, 384, 32):
                self.reshape_recordings(time_interval=interval, electrode_range=(0, 100000))
                self.correlation(electrode=i,
                                 title=f'corrcoef between electrode {i} to all electrodes at time {interval}',
                                 path=path)
        if should_zip:
            shutil.make_archive(dir_name, 'zip', dir_name)

    def correlation(self, electrode, title, path):
        corr = [np.corrcoef(np.transpose(self.reshaped_recordings)[electrode], record)[0][1]
                for record in np.transpose(self.reshaped_recordings)]
        plot_and_save(range(384), corr, title, f'{path}/{title}.png')

    def var_with_window(self, window=0.1, should_zip=False):
        path = f'output/var on all electrodes with window'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        for time in range(60, 360, 60):
            interval = (time, time + window)
            self.reshape_recordings(time_interval=interval, electrode_range=(0, 100000))
            self.var(title=f'var on all electrodes with time at time: {interval}', path=path)

        if should_zip:
            shutil.make_archive(path, 'zip', path)

    def var(self, title, path):
        var = [np.var(record) for record in np.transpose(self.reshaped_recordings)]
        plot_and_save(range(384), var, title, f'{path}/{title}.png')

    def avg_with_window(self, window):
        self.avg_recordings = np.array([avg(record, window) for record in np.transpose(self.reshaped_recordings)])

    def kmeans(self, data=None, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters)
        data = data if data else self.reshaped_recordings
        kmeans.fit(np.transpose(data))
        self.centroids, self.clusters = kmeans.cluster_centers_, kmeans.labels_

    def active_times(self, threshold=450):  # find times when record was greater then threshold
        times_when_greater = []
        for electrode in range(0, 384):
            times_when_greater.append(
                self.times_when_greater_then_threshold_for_electrode(electrode=electrode, threshold=threshold))
        self.show_table(data=times_when_greater)

    def times_when_greater_then_threshold_for_electrode(self, threshold: int = 400, max_time=100, electrode=0):
        times_where_greater_then_threshold = []
        for i in range(0, max_time):
            time_interval = (i, i + 1)
            self.reshape_recordings(time_interval=time_interval, electrode_range=(electrode, electrode + 1))
            for magnitude in self.reshaped_recordings:
                if abs(magnitude) >= threshold:
                    times_where_greater_then_threshold.append(time_interval)
                    break
        return times_where_greater_then_threshold

    def find_maxs_for_electrode_and_time_window(self, electrode, window):
        maxima = []
        for i in range(0, int(100 / window)):
            time = (i * window, (i + 1) * window)
            self.reshape_recordings(time_interval=time, electrode_range=(electrode, electrode + 1))
            maximum = np.max(np.absolute(self.reshaped_recordings))
            maxima.append(maximum)
        return maxima

    def find_maxs_for_all_electrodes_and_single_time(self):
        maxima = []
        for i in range(0, 100, 5):
            time = (i, i + 1 / 30000)
            self.reshape_recordings(time_interval=time, electrode_range=(0, 384))
            maximum = np.reshape(np.array(self.reshaped_recordings), (12, 32))
            maximum = np.array([np.max(np.absolute(record)) for record in maximum]).flatten()
            maxima.append(maximum)
        self.show_table(maxima)
        return maxima

    def find_times_when_records_was_above_min_threashold(self):
        min = np.min([np.max(np.absolute(record)) for record in np.transpose(self.reshaped_recordings)])
        maxima = [np.argwhere(np.absolute(record) >= min) for record in np.transpose(self.reshaped_recordings)]
        times = [_find_continuous_times_when_biger_then_threshold_for_record(maximums) for maximums in maxima]
        self.show_table(data=times)

    def plot_trajectory_for_electrode_at_time(self, electrode, time):
        title = f'Trajectory of the electrode {electrode} at times {time}'
        plot_and_save(range(3000000),
                      np.transpose(self.reshaped_recordings)[electrode - 1][time[0]: time[1]],
                      title,
                      f'output/{title}')


if __name__ == '__main__':
    recordings = Recordings()
    recordings.main()
