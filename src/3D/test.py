import numpy as np

from mesh import plot_mesh
from optimization_3d import find_droplet_optimization_3d

gamma = 0.0728      #N/m
density = 1000.0    #kg/m³
a = 0.001           # raio do bico circular (m)
g = 9.81
Bond = a**2 * density * g / gamma

V_adim = 10.0

# bico circular de raio 1: o anel do corte da calota é a borda do bico
d = find_droplet_optimization_3d(V_adim, Bond, 12, 24, "circle", QuadRule="strang3")
plot_mesh(d)

# mesma gota com uma parede plana em x = 1, encostada no bico
d = find_droplet_optimization_3d(V_adim, Bond, 12, 24, "circle", wall=1.0)
plot_mesh(d)

# bico quadrado [-1, 1]^2 (um colar de triângulos liga a calota ao quadrado)
nozzle = np.array([
    [1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0],
    [-1.0, 1.0]
])
d = find_droplet_optimization_3d(V_adim, Bond, 12, 24, nozzle)
plot_mesh(d)
