import numpy as np
import scipy.optimize as optm

from mesh import *
from quadrature_triangle import *


def find_droplet_optimization_3d(droplet_Volume:float, Bond: float, num_stacks: int, num_sectors:float, nozzle:np.array,  constraints: List = []) -> Droplet:
    mesh = Mesh.cut_sphere(
        r=1.0,
        num_sectors=num_sectors,
        num_stacks=num_stacks,
        min_theta=np.pi / 2.0,
        z_center= -0.05
    )

    center = point(0.0, 0.0, -0.05)
    N = len(mesh.points)
    x0 = np.full(N, 0.0)
    
    mesh.attach_nozzle(nozzle, 0.0, 0.0)
    v = lambda x: volume_3D(x, mesh, center)

    energy = lambda x: energy_3D(x, mesh, Bond, center)

    volume_constraint = optm.NonlinearConstraint(v, droplet_Volume, droplet_Volume)

    constraints.append(volume_constraint)
    lb = np.full(N, -1.0 + 1.0e-8)
    ub = np.full(N,  np.inf)

    bound = optm.Bounds(lb, ub)
    res = optm.minimize(
        energy, 
        x0, 
        method='SLSQP', 
        constraints=constraints, 
        bounds=bound, 
        options={
            'maxiter': 10000,
            'ftol': 1e-12,
            'disp': True
        }
    )
    print("Success:", res.success)
    print("Message:", res.message)
    print("Iterations:", res.nit)
    print("Energy:", res.fun)

    print("Target volume:", droplet_Volume)
    print("Computed volume:", volume_3D(
        res.x, mesh, center
    ))

    return deformed_mesh(res.x, mesh, center)
