import math


class ImprovedEulerMethod:

    def __init__(self, f, y_0, x_0, x_end, steps):
        dx = (x_end - x_0) / steps

        x = x_0
        y = y_0

        self.values = [y_0]

        for i in range(1, steps):
            try:
                f_0 = f(y, x)
                k1 = f_0 * dx
                y_predict = y + k1
                f_1 = f(y_predict, x + dx)
                y = y + (dx / 2) * f_0 + (dx / 2) * f_1
                x = x + dx
                self.values.append(y)

            except OverflowError:
                self.values.append(math.inf)

