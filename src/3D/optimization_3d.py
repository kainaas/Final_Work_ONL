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


    def volume(h):
        return np.pi * (
            h + 2/3 * (
                h**3 + (1 + h**2)**1.5
            )
        )
    
    def dvolume(h):
        return np.pi * (
            np.sqrt(1 + h**2) + h
        )**2

    def sphere_from_volume(V0, h0=0.0):
        h = optm.newton(
            lambda h: volume(h) - V0,
            x0=h0,
            fprime=dvolume,
            tol=1e-12
        )
        R = np.sqrt(1 + h**2)
        return R, h

    R,h = sphere_from_volume(droplet_Volume)


    center = point(0.0, 0.0, -h)

    mesh = Mesh.cut_sphere(
        r=R,
        num_sectors=num_sectors,
        num_stacks=num_stacks,
        min_theta=np.arccos(h/R),
        z_center=center[2] - R/4.0
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
