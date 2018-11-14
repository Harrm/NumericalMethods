import math


class EulerApproximation:

    def __init__(self, y_0, x_0, x_f, f, steps=1000):
        self.f = f
        self.values = [y_0]
        self.x_start = x_0
        self.x_end = x_f
        self.steps = steps

        dx = (self.x_end - self.x_start) / self.steps
        x = self.x_start
        for i in range(1, self.steps):
            try:
                y_prime = self.f(self.values[i-1], x)
                self.values.append(self.values[i-1] + y_prime * dx)
            except OverflowError:
                print("x: ")
                self.values.append(math.inf)
            x += dx
