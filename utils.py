import numpy as np
import matplotlib.pyplot as plt


def avg(record, window):
    return np.array([sum(record[i: i + window]) / window for i in range(0, len(record), window)])


def plot_and_save(x, y, title, file_path):
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(file_path)
    plt.show()


def _find_continuous_times_when_biger_then_threshold_for_record(times_in_single_record):
    intervals: list[tuple] = []
    start = times_in_single_record[0]
    end = times_in_single_record[0]
    count = 0
    last_iter_at_if = True
    for i in range(len(times_in_single_record)):
        if start + count == times_in_single_record[i]:
            count += 1
            end = times_in_single_record[i]
            last_iter_at_if = True
        else:
            interval = (start, end)
            intervals.append(interval)
            start = end = times_in_single_record[i]
            count = 1
            last_iter_at_if = False
    if last_iter_at_if:
        interval = (start, end)
        intervals.append(interval)

    return intervals
