from typing import List, Optional
import numpy as np
import scipy.optimize as optm

from mesh_quad import QuadMesh, deformed_quad_mesh, point
from quadrature_quad import energy_3D_quad, volume_3D_quad


def find_droplet_optimization_3d_quad(
    droplet_Volume: float,
    Bond: float,
    num_stacks: int,
    num_sectors: int,
    nozzle: np.ndarray,
    constraints: Optional[List] = None,
    fix_boundary: bool = False,
    n_gauss: int = 3,
) -> QuadMesh:
    # lista criada aqui dentro: evita o bug do argumento mutavel `= []`
    constraints = [] if constraints is None else list(constraints)

    center = point(0.0, 0.0, -0.2)
    mesh = QuadMesh.cut_sphere(
        r=1.0, num_sectors=num_sectors, num_stacks=num_stacks,
        min_theta=np.pi / 2.0, z_center=center[2])

    N = len(mesh.points)          # so os vertices da esfera sao variaveis
    x0 = np.zeros(N)
    mesh.attach_nozzle(nozzle, 0.0, 0.0)

    v = lambda x: volume_3D_quad(x, mesh, center, n_gauss)
    e = lambda x: energy_3D_quad(x, mesh, Bond, center, n_gauss)
    constraints.append(optm.NonlinearConstraint(v, droplet_Volume, droplet_Volume))

    lb = np.full(N, -1.0 + 1.0e-8)
    ub = np.full(N, np.inf)
    if fix_boundary:              # prende a linha de contato (anel do corte)
        lb[mesh.cut_boundary] = 0.0
        ub[mesh.cut_boundary] = 0.0

    res = optm.minimize(e, x0, method="SLSQP", constraints=constraints,
                        bounds=optm.Bounds(lb, ub),
                        options={"maxiter": 10000, "ftol": 1e-12, "disp": True})

    print("Success:", res.success, "|", res.message)
    print("Iterations:", res.nit, " Energy:", res.fun)
    print("Target volume:", droplet_Volume, " Computed:", v(res.x))
    return deformed_quad_mesh(res.x, mesh, center)
