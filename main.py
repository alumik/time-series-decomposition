import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Sequence, Tuple
from statsmodels.tsa.seasonal import STL
from pandas.plotting import register_matplotlib_converters


def linear_interpolation(timestamps: Sequence, values: Sequence) -> Tuple[np.ndarray, np.ndarray, int]:
    timestamps = np.asarray(timestamps, np.int64)
    values = np.asarray(values)

    src_index = np.argsort(timestamps)
    timestamp_sorted = timestamps[src_index]
    intervals = np.unique(np.diff(timestamp_sorted))
    interval = np.min(intervals)
    if interval == 0:
        raise ValueError('Duplicated values in `timestamp`')
    for itv in intervals:
        if itv % interval != 0:
            raise ValueError('Not all intervals in `timestamp` are multiples '
                             'of the minimum interval')

    length = (timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1
    ret_timestamp = np.arange(timestamp_sorted[0],
                              timestamp_sorted[-1] + interval,
                              interval,
                              dtype=np.int64)
    missing = np.ones(length, dtype=np.int32)
    ret_values = np.zeros(length, dtype=values.dtype)
    dst_index = np.asarray((timestamp_sorted - timestamp_sorted[0]) // interval, dtype=np.int)
    missing[dst_index] = 0
    ret_values[dst_index] = values[src_index]
    miss_index = np.argwhere(missing == 1).reshape(-1)

    if len(miss_index) > 0:
        pos_index = np.argwhere(missing == 0).reshape(-1)
        pos_values = ret_values[pos_index]
        ret_values[miss_index] = np.interp(miss_index, pos_index, pos_values)

    return ret_timestamp, ret_values, interval


def stl_and_plot(file: str, seconds: int = 1_500_000):
    df = pd.read_csv(os.path.join('data', file))
    timestamps, values, interval = linear_interpolation(df.timestamp, df.value)
    values = (values - values.mean()) / values.std()
    timestamps = timestamps[:seconds // interval]
    values = values[:seconds // interval]

    start_time = datetime.datetime.fromtimestamp(timestamps[0])
    end_time = datetime.datetime.fromtimestamp(timestamps[-1])
    series = pd.Series(values,
                       index=pd.date_range(start=start_time, end=end_time, freq=pd.offsets.Second(interval)),
                       name=file)

    stl = STL(series, period=24 * 3600 // interval, seasonal=21)
    res = stl.fit()
    res.plot()
    plt.savefig(os.path.join('out', f'{file}.png'))


def main():
    register_matplotlib_converters()
    sns.set_style('darkgrid')
    plt.rc('figure', figsize=(16, 12))
    plt.rc('font', size=13)

    os.makedirs('out', exist_ok=True)
    files = os.listdir('data')
    for file in files:
        print(f'Processing {file}...')
        stl_and_plot(file)


if __name__ == '__main__':
    main()
