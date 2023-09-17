import shutil
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from utils import (DATA_FILE_PATH,
                   _find_continuous_times_when_biger_then_threshold_for_record,
                   plot_and_save,
                   avg)

CORRELATION_ELECTRODES = [40, 50, 60, 70, 180, 190, 200, 210, 300, 310, 320, 330]


class Recordings:
    parsed_recordings: np.ndarray
    reshaped_recordings: np.ndarray
    avg_recordings: np.ndarray
    clusters: np.ndarray
    centroids: np.ndarray
    classifaier: LogisticRegression

    '''Processing'''

    def main(self):
        self.parse_recordings()
        self.linear_classifaier()

    def parse_recordings(self):
        with open(file=DATA_FILE_PATH, mode='r') as file:
            parsed_recordings = np.fromfile(file=file, dtype=np.int16)
            parsed_recordings = parsed_recordings.reshape(int(parsed_recordings.shape[0] / 385), 385)
        self.parsed_recordings = parsed_recordings[:, :-1]
        self.reshape_recordings()

    def reshape_recordings(self, time_interval=(0, 10800000), electrode_range=(0, 385)):
        self.reshaped_recordings = self.parsed_recordings[int(time_interval[0] * 30000): int(time_interval[1] * 30000),
                                   int(electrode_range[0]): int(electrode_range[1])]

    def show_table(self, data=None):
        data = self.reshaped_recordings if data is None else data
        df = pd.DataFrame(data)
        print(df)

    def plot_trajectory_for_electrode_at_time(self, electrode, time):
        title = f'Trajectory of the electrode {electrode} at times {time}'
        plot_and_save(range(time[0], time[1]),
                      np.transpose(self.reshaped_recordings)[electrode - 1][time[0]: time[1]],
                      title,
                      f'output/{title}')

    '''Learning Models:'''

    def kmeans(self, n_clusters=3, should_use_pca=False):
        self.reshape_recordings(time_interval=(60, 360))
        data = np.corrcoef(np.transpose(self.reshaped_recordings))
        kmeans = KMeans(n_clusters=n_clusters)
        pd.DataFrame(data).to_csv("corr matrix.csv")
        kmeans.init = [data[70][:], data[200][:], data[320][:]]
        kmeans.fit(data)
        self.centroids, self.clusters = kmeans.cluster_centers_, kmeans.labels_
        self.show_table(data=self.clusters)
        clusters = pd.DataFrame(np.transpose(self.clusters))
        centroids = pd.DataFrame(np.transpose(self.centroids))
        clusters.to_csv(path_or_buf=f'output/with init on all time kmeans with {n_clusters} clusters: clusters.csv')
        centroids.to_csv(path_or_buf=f'output/with init on all time kmeans with {n_clusters} clusters: centroids.csv')

    def correlation(self, electrode=0, title='', path='', should_plot=False):
        print(f'electrode: {electrode}')
        corr = [np.corrcoef(np.transpose(self.reshaped_recordings)[electrode],
                            np.transpose(self.reshaped_recordings)[index])[0][1]
                for index in CORRELATION_ELECTRODES]
        if should_plot:
            plot_and_save(range(384), corr, title, f'{path}/{title}.png')
        return corr

    def linear_classifaier(self):
        a, b, c = np.full((70), 0), np.full((130), 1), np.full((114), 2)
        target = np.concatenate((a, b, c), axis=0)
        self.reshape_recordings(time_interval=(100, 150))
        data = np.concatenate((np.transpose(self.reshaped_recordings)[30: 100], np.transpose(self.reshaped_recordings)[120: 250], np.transpose(self.reshaped_recordings)[270: 384]), axis=0)
        lr = LogisticRegression()
        lr.fit(data, target)
        self.classifaier = lr
        self.reshape_recordings(time_interval=(100, 150))
        test = np.transpose(self.reshaped_recordings)
        pd.DataFrame(self.classifaier.predict(test)).to_csv(f'linear classifiers lables.xlsx')
        first, second = 0, 0
        prev = 0
        for index, val in enumerate(self.classifaier.predict(test)):
            if index == 0:
                continue
            if val - prev == 1:
                if first != 0:
                    second = index
                else:
                    first = index
            prev = val
        print(f'[0, {first}] \n[{first + 1}, {second}] \n[{second + 1}, 384]')

    '''Helper Functions for Understanding and Cleaning the Data'''

    def correlation_with_window(self, window=0.1, should_plot=False, should_save=False, should_zip=False):
        dir_name = 'output/corrcoef on single electrode with all electrodes with window'
        if should_save:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            os.makedirs(dir_name)

        for time in range(60, 360, 60):
            interval = (time, time + window)
            path = f'{dir_name}/corrcoef on single electrode with all electrodes with window at time {interval}'
            if should_save:
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.makedirs(path)
            for i in range(0, 384, 32):
                self.reshape_recordings(time_interval=interval, electrode_range=(0, 100000))
                self.correlation(electrode=i,
                                 title=f'corrcoef between electrode {i} to all electrodes at time {interval}',
                                 path=path)
        if should_save and should_zip:
            shutil.make_archive(dir_name, 'zip', dir_name)

    def var_with_window(self, window=0.1, should_plot=False, should_save=False, should_zip=False):
        path = f'output/var on all electrodes with window'
        if should_save:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        for time in range(60, 360, 60):
            interval = (time, time + window)
            self.reshape_recordings(time_interval=interval, electrode_range=(0, 100000))
            self.var(title=f'var on all electrodes with time at time: {interval}', path=path)

        if should_save and should_zip:
            shutil.make_archive(path, 'zip', path)

    def avg_with_window(self, window):
        self.avg_recordings = np.array([avg(record, window) for record in np.transpose(self.reshaped_recordings)])

    def var(self, title='', path='', should_plot=False):
        var = [np.var(record) for record in np.transpose(self.reshaped_recordings)]
        if should_plot:
            plot_and_save(range(384), var, title, f'{path}/{title}.png')

    def active_times(self, threshold=450):
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


if __name__ == '__main__':
    recordings = Recordings()
    recordings.main()
