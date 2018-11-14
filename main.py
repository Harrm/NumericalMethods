import matplotlib.pyplot as plot
import numpy
import euler_approximation as ea
import runge_kutta as rk
import improved_euler as ie
import math

f = lambda y, x: (x * (y ** 1.5) + x * y)

y_0 = 4
x_0 = 1
x_end = 7.8
steps = 1000

appr1 = ea.EulerApproximation(y_0, x_0, x_end, f, steps)
appr2 = rk.RungeKutta(f, x_0, y_0, x_end, steps)
appr3 = ie.ImprovedEulerMethod(f, y_0, x_0, x_end, steps)
x = numpy.linspace(x_0, x_end, steps)


f_exact = lambda x: (4 * numpy.exp(x*x / 2)) / (math.e**(0.25) - 2*numpy.exp(x*x / 4))**2


def plot_approxs(plotter):
    plotter.plot(x, f_exact(x), label='Exact solution')
    plotter.plot(x, appr1.values, label='Euler approximation')
    plotter.plot(x, appr2.values, label='Runge-Kutta approximation', alpha=0.5)
    plotter.plot(x, appr3.values, label='Improved Euler approximation', alpha=0.5, color='blue')


def plot_error(plotter):
    err1 = [0]*990
    err2 = [0]*990
    err3 = [0]*990
    for i in range(10, 1000, 10):
        e = ea.EulerApproximation(y_0, x_0, x_end, f, i).values
        err1[i-10] = abs(e[-1] - f_exact(x_end))
        m = ie.ImprovedEulerMethod(f, y_0, x_0, x_end, i).values
        err2[i-10] = abs(m[-1] - f_exact(x_end))
        r = rk.RungeKutta(f, x_0, y_0, x_end, i).values
        err3[i-10] = abs(r[-1] - f_exact(x_end))
        print(e[-1], m[-1], r[-1])

    i = numpy.linspace(10, 1000, 990)
    plotter.axis([10, 1000, 0, 1000])
    plotter.plot(i, err1, label='Euler approximation error')
    plotter.plot(i, err2, label='Improved Euler approximation error')
    plotter.plot(i, err3, label='Runge-Kutta approximation error', alpha=0.5)


plot.style.use('seaborn-dark')

ax = plot.subplot(2, 1, 1)
ax.axis([0, x_end, 0, 10])
plot_approxs(ax)

plot.xlabel('x label')
plot.ylabel('y label')

ax.set(title="Solution Plot")

ax.legend()
ax.grid()

ax1 = plot.subplot(2, 1, 2)
ax1.axis([0, steps, 0, 1])
plot_error(ax1)

plot.xlabel('steps')
plot.ylabel('error')

ax1.legend()
ax1.grid()

plot.show()