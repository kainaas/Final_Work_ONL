from sys import path

path.append("./../src/2D")

import numpy as np
import matplotlib.pyplot as plt

from droplet_optimization import *
from droplet import *


gamma = 0.0728      #N/m
density = 1000.0    #kg/m³
a = 0.001
Bond = a**2 * density * const.g / gamma

V_adim = 1.0
print("Initiating")

A: float
if V_adim <= 2.17221828749:
    A = shooting(V_adim, 0.0, 2.0, Bond, a)
elif V_adim > 20.0:
    A = shooting(V_adim, 1.2, 1.4, Bond, a)
elif V_adim > 10.0:
    A = shooting(V_adim, 1.3, 1.6, Bond, a)
elif V > 5.0:
    A = shooting(V_adim, 1.5, 1.9, Bond, a)
elif v_adim > 2.17221828749:
    A = shooting(V_adim, 1.2, 1.9, Bond, a)

ld = find_droplets(A, Bond, a, np.pi * 5.0, 0.001)
if len(ld) >= 2:
    ld = ld[:2]
drop_shooting = sorted(ld, key=lambda d: abs(d.V - V_adim))[0]

print("Shooting done")
drop_optm = find_droplet_optimization(V_adim, a, Bond, 100)
print("Optimization done")

fig, ax = plt.subplots(2,2)
ax[0,0].set_aspect("equal")
ax[0,1].set_aspect("equal")

ax[1, 1].set_aspect("equal")


drop_shooting.plot(ax[0,0], label="shooting")
ax[0,0].set_title("Shooting")

drop_optm.plot(ax[0,1], label="optimization")
ax[0,1].set_title("optimization")

drop_shooting.plot(ax[1,1], label="shooting")
drop_optm.plot(ax[1,1], label="optimization")
ax[1,1].legend()
plt.show()
