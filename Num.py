import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def predator_prey(y, t, a, b, c, d, k, A):
    X, Y = y
    dXdt = a * X *(1-X/k) - ((b*X*Y)/(1+A*X))
    dYdt = -c*Y + ((d*X*Y)/(1+A*X))
    return [dXdt, dYdt]


a = 1.0
b = 0.1
c = 1.5
d = 0.075
k = 2
A = 1

# Начальные условия
X0 = 10.0
Y0 = 5.0
y0 = [X0, Y0]

# Временные точки для интеграции
t = np.linspace(0, 10, 100)

# Решение системы уравнений
sol = odeint(predator_prey, y0, t, args=(a, b, c, d, k, A))

# График популяций
plt.plot(t, sol[:, 0], 'b', label='Жертвы')
plt.plot(t, sol[:, 1], 'r', label='Хищники')
plt.xlabel('Время')
plt.ylabel('Популяция')
plt.title('Модель хищник-жертва')
plt.legend()
plt.show()
