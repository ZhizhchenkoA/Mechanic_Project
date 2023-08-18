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


x_diff = derivative(f)

x_plt = np.arange(0, 5, 0.1)
y_plt = [f(x) for x in x_plt]
plt.ion()
fig, ax = plt.subplots()
ax.grid = True
ax.plot(x_plt, y_plt)
N = 200  # число итераций
xx = 0  # начальное значение
lmd = 0.9  # шаг сходимости


def gradient_one_var(N, xx, lmd, x_diff=x_diff, f=f):
    point = ax.scatter(xx, f(xx), color='red')

    for i in range(N):
        xx -= lmd * x_diff(xx)
        point.set_offsets([xx, f(xx)])

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.02)
    print(xx)


gradient_one_var(N, xx, lmd)
