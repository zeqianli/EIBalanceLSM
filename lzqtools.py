"""Some simple tools.

Author: Zeqian Li
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import json, os, pickle


# Plot tools

def plot_1d(arr, **kwargs):
    return plt.plot(range(len(arr)), arr, **kwargs)


# File tools

def save_json(data, path, mode='w', inspect=True):
    with open(path, mode) as f:
        try:
            json.dump(data, f)
        except TypeError:
            if type(data) is np.ndarray:
                data = data.tolist()
            elif type(data) is dict:
                for key, value in data.items():
                    if type(value) is np.ndarray:
                        data[key] = value.tolist()
                else:
                    raise TypeError
            json.dump(data, f)
    if inspect:
        print("data saved at %s, mode=%s" % (path, mode))


def load_json(path, mode='r', inspect=True):
    with open(path, mode) as f:
        data = json.load(f)
    if inspect:
        print("data loaded at %s" % path)
    return data


def save_pickle(data, path, mode='wb', inspect=True):
    with open(path, mode) as f:
        pickle.dump(data, f)

    if inspect:
        print("data saved at %s, mode=%s" % (path, mode))


def load_pickle(path, mode='rb', inspect=True):
    with open(path, mode) as f:
        data = pickle.load(f)
    if inspect:
        print("data loaded at %s" % path)
    return data


def localtime_str(fmt='%.4d%.2d%.2d_%.2d%.2d%.2d'):
    from time import localtime
    tm = localtime()
    return fmt % (tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)


def date_str(fmt='%.2d.%.2d'):
    from time import localtime
    tm = localtime()
    return fmt % (tm.tm_mon, tm.tm_mday)


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        pass