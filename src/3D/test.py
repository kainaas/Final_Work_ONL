import numpy as np

from mesh import plot_mesh
from optimization_3d import find_droplet_optimization_3d

gamma = 0.0728      #N/m
density = 1000.0    #kg/m³
a = np.sqrt(2) * 0.001
g = 9.81
Bond = a**2 * density * g / gamma

V_adim = 1.0

nozzle = np.array([
    [1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0],
    [-1.0, 1.0]
])
# fix_boundary=True keeps the contact line (spherical cut ring) fixed
d = find_droplet_optimization_3d(V_adim, Bond, 8, 16, nozzle, fix_boundary=False, QuadRule="strang3")
plot_mesh(d)
