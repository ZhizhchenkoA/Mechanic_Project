import matplotlib.pyplot as plt
import numpy as np
import tqdm
import time


def f(x):
    return x ** 2 - 5 * x + 5


def derivative(func):
    epsilon = 0.000001

    def wrapper(x):
        return (f(x + epsilon) - f(x)) / epsilon

    return wrapper


def gradient_one_var(N, xx, lmd, x_diff, f=f):
    fig, ax = plt.subplots()

    point = ax.scatter(xx, f(xx), color='red')

    for i in range(N):
        xx -= lmd * x_diff(xx)
        point.set_offsets([xx, f(xx)])

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.02)
    print(xx)


