import numpy as np
import scipy.optimize as optm

from mesh import Mesh, deformed_mesh, point
from quadrature_triangle import energy_3D, volume_3D


def find_droplet_optimization_3d(
    droplet_Volume: float,
    Bond: float,
    num_stacks: int,
    num_sectors: int,
    nozzle: np.ndarray,
    constraints: list | None = None,
    fix_boundary: bool = False,
    QuadRule: str = "strang3",
    verbose: bool = True,
) -> Mesh:
    '''
        Minimizes the energy of a triangulated droplet hanging from a nozzle of
        arbitrary polygonal shape, with fixed volume (SLSQP).

        fix_boundary: if True, the vertices of the spherical cut (contact line)
        are kept fixed; otherwise they move radially like the other vertices.
        QuadRule: triangle quadrature rule ("strang3", "strang7" or "strang10").
    '''
    # new list on every call: the caller's list is not modified and the
    # volume constraint does not accumulate between calls
    constraints = list(constraints or [])

    center = point(0.0, 0.0, -0.2)
    mesh = Mesh.cut_sphere(
        r=1.0,
        num_sectors=num_sectors,
        num_stacks=num_stacks,
        min_theta=np.pi / 2.0,
        z_center=center[2]
    )

    # only the spherical vertices are variables; the nozzle ring is fixed
    N = len(mesh.points)
    x0 = np.zeros(N)

    mesh.attach_nozzle(nozzle, 0.0, 0.0)
    v = lambda x: volume_3D(x, mesh, center, QuadRule)
    energy = lambda x: energy_3D(x, mesh, Bond, center, QuadRule)

    constraints.append(optm.NonlinearConstraint(v, droplet_Volume, droplet_Volume))

    lb = np.full(N, -1.0 + 1.0e-8)
    ub = np.full(N, np.inf)
    if fix_boundary:
        lb[mesh.cut_boundary] = 0.0
        ub[mesh.cut_boundary] = 0.0

    res = optm.minimize(
        energy,
        x0,
        method='SLSQP',
        constraints=constraints,
        bounds=optm.Bounds(lb, ub),
        options={
            'maxiter': 10000,
            'ftol': 1e-12,
            'disp': verbose
        }
    )
    if verbose:
        print("Success:", res.success)
        print("Message:", res.message)
        print("Iterations:", res.nit)
        print("Energy:", res.fun)
        print("Target volume:", droplet_Volume)
        print("Computed volume:", v(res.x))

    return deformed_mesh(res.x, mesh, center)
