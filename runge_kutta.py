import math

class RungeKutta:

    def __init__(self, f, x_0, y_0, x_end, steps):
        self.values = [0] * steps
        dx = (x_end - x_0) / steps
        self.values[0] = y_0
        x = x_0
        y = y_0
        for i in range(1, steps):
            try:
                k1 = dx * f(y, x)
                k2 = dx * f(y + 0.5 * k1, x + 0.5 * dx)
                k3 = dx * f(y + 0.5 * k2, x + 0.5 * dx)
                k4 = dx * f(y + k3, x + dx)
                x = (x_0 + i * dx)
                self.values[i] = y = y + (k1 + k2 + k2 + k3 + k3 + k4) / 6
            except OverflowError:
                self.values[i] = math.inf
