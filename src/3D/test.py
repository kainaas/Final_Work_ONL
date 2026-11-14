from mesh import *
from quadrature_triangle import *
from optimization_3d import *

gamma = 0.0728      #N/m
density = 1000.0    #kg/m³
a = 0.001
g = 9.81
Bond = a**2 * density * g / gamma

V_adim = 20.0

nozzle = np.array([
    [1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0],
    [-1.0, 1.0]
])
d = find_droplet_optimization_3d(V_adim, Bond, 8, 16, nozzle)
plot_mesh(d)