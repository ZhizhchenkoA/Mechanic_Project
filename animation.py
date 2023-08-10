import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

mu = 0.2  # приведённое отношение масс объектов


def f(x, y):
    """Функция U(x, y)"""
    r1 = ((x + mu) ** 2 + y ** 2) ** 0.5
    r2 = ((x - 1 + mu) ** 2 + y ** 2) ** 0.5
    return (x ** 2 + y ** 2) / 2 + (1 - mu) / r1 + mu / r2 + ((1 - mu) * mu) / 2


def ro(k, x, y, px, py):
    """Линеаризация значений в коллинеарных точках Лагрнжа"""
    ro = mu * abs(k - 1 + mu) ** (-3) + (1 - mu) * abs(k + mu) ** (-3)
    a = 2 * ro + 1
    b = ro - 1
    return 0.5 * ((px + y) ** 2 + (py - x) ** 2 - a * x ** 2 + b * y ** 2)


def h(x, y, px, py):
    """Гамильтониан системы"""
    return ((px + y) ** 2 + (py - x) ** 2) / 2 - f(x, y)


def diff_u_x(x, y):
    """Производная в точке от U(x, y) по x"""
    delta = 0.00001
    return (-1) * (f(x + delta, y) - f(x, y)) / delta


def diff_u_y(x, y):
    """Производная в точке от U(x, y) по y"""
    delta = 0.00001
    return (-1) * (f(x, y + delta) - f(x, y)) / delta


def x_dot(x, y, px, py):
    """Скорость по x"""
    return px + y


def y_dot(x, y, px, py):
    """Скорость по y"""
    return py - x


def px_dot(x, y, px, py):
    """Производная по px"""
    return py - x + diff_u_x(x, y)


def py_dot(x, y, px, py):
    """Производная по py"""
    return -px - y + diff_u_y(x, y)


def W(x, y):
    """Функция нормализации значения"""
    return -math.log((abs(diff_u_x(x, y)) + abs(diff_u_y(x, y))) ** 0.01 + 1e-10, 10)


def diff_h_px(x, y, px, py):
    """Производная в точке от H(x, y, px, py) по px"""
    delta = 0.0001
    return (h(x, y, px + delta, py) - h(x, y, px, py)) / delta


def diff_h_py(x, y, px, py):
    """Производная в точке от H(x, y, px, py) по py"""
    delta = 0.0001
    return (h(x, y, px, py + delta) - h(x, y, px, py)) / delta


def h_const(x, y, px=0, py=0):
    """Подсчёт постоянной Якоби"""
    return 0.5 * (x_dot(x, y, px, py) ** 2 + y_dot(x, y, px, py) ** 2) - f(x, y)


class FloatRange:
    """Класс для использования range с нецелыми числами"""

    def __init__(self, start, stop, step=1.0):
        if not isinstance(start, (int, float)) or \
                not isinstance(stop, (int, float)) or not isinstance(step, (int, float)):
            raise TypeError('start, stop, step должны быть числами!')
        if start > stop and step < 0 or stop > start and step > 0:
            self.start = float(start)
            self.stop = stop
            self.step = step
            self.value = self.start - self.step
        else:
            raise ValueError('введите корректные значения')

    def __iter__(self):
        self.value = self.start
        while self.value < self.stop and self.step > 0 or self.value > self.stop and self.step < 0:
            yield self.value
            self.value += self.step
        return self

    def __len__(self):
        return int(abs(abs(self.start - self.stop) / self.step))


def frange(start, stop, step=1.0):
    """Функция для создания итератора на основе FloatRange"""
    return FloatRange(start, stop, step)


from celluloid import Camera

fig = plt.figure()
camera = Camera(fig)


def move_animation_rotating(x, y, px=0, py=0, t=100000, scale_large=False):
    """Создание анимации на основе движения третьего тела в синодической системе отсчёта"""
    delta_t = 0.001
    X, Y, PX, PY = [x], [y], [px], [py]
    for i in tqdm(range(t)):
        X.append(X[-1] + x_dot(x, y, px, py) * delta_t)
        Y.append(Y[-1] + y_dot(x, y, px, py) * delta_t)
        PX.append(PX[-1] + px_dot(x, y, px, py) * delta_t)
        PY.append(PY[-1] + py_dot(x, y, px, py) * delta_t)
        x = X[-1]
        y = Y[-1]
        px = PX[-1]
        py = PY[-1]
        if i % 1000 == 0:
            plt.scatter([-mu, 1 - mu], [0, 0], color='red', )
            plt.plot(X, Y, color='blue')

            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            camera.snap()
    return (X, Y)


def move_animation_non_rotating(x, y, px=0, py=0, delta_t=0.001, t=100000):
    ro1, ro2 = -mu, 1 - mu
    psi = 0
    X, Y, PX, PY = [x], [y], [px], [py]

    for i in tqdm(range(t)):
        X[-1] = x + x_dot(x, y, px, py) * delta_t
        Y[-1] = y + y_dot(x, y, px, py) * delta_t
        PX[-1] = px + px_dot(x, y, px, py) * delta_t
        PY[-1] = py + py_dot(x, y, px, py) * delta_t
        x = X[-1]
        y = Y[-1]
        px = PX[-1]
        py = PY[-1]
        psi += delta_t
        x1, y1 = ro1 * math.cos(psi), ro1 * math.sin(psi)
        x2, y2 = ro2 * math.cos(psi), ro2 * math.sin(psi)
        if i % 100 == 0:
            circle = np.arange(0, 2 * np.pi + 0.2, 0.1)
            plt.plot(np.cos(circle) * ro1, np.sin(circle) * ro1, color='black')
            plt.plot(np.cos(circle) * ro2, np.sin(circle) * ro2, color='black')

            x_1 = x * math.cos(psi) - y * math.sin(psi)
            y_1 = x * math.sin(psi) + y * math.cos(psi)
            plt.scatter([x_1], [y_1], color='magenta', linewidths=0.5)
            plt.scatter([x1, x2], [y1, y2], color='red')
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            camera.snap()


move_animation_non_rotating(0.5, 0.5)
animation = camera.animate(interval=100)
animation.save('gif.gif', writer='pillowwriter', fps=15)
plt.show()
