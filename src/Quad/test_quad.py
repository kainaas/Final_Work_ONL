import numpy as np
from mesh_quad import *
from optimization_3d_quad import find_droplet_optimization_3d_quad

gamma, density, g = 0.0728, 1000.0, 9.81
a = np.sqrt(2) * 0.001
Bond = (a**2 * density * g / gamma)+0.1

V_adim = 15

nozzle = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
d = find_droplet_optimization_3d_quad(V_adim, Bond, 16, 32, nozzle, fix_boundary=True)
plot_mesh(d)
