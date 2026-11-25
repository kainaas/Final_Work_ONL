import numpy as np
import scipy.optimize as optm

from mesh import Mesh, deformed_mesh, point
from quadrature_triangle import energy_3D, volume_3D, grad_energy_3D, grad_volume_3D


# ---------------------------------------------------------------------------
# initial shape: a spherical cap whose (discrete) volume is the droplet volume
# ---------------------------------------------------------------------------

def cap_volume(h):
    '''
        Volume below the plane z = 0 of the sphere centred at (0, 0, -h) that
        passes through the unit circle of that plane (R = sqrt(1 + h^2)).
    '''
    return np.pi * (h + 2/3 * (h**3 + (1 + h**2)**1.5))


def dcap_volume(h):
    return np.pi * (np.sqrt(1 + h**2) + h)**2


def sphere_from_volume(V0, h0=0.0):
    '''
        (R, h) of the spherical cap of volume V0 (exact formula, Newton).
    '''
    h = optm.newton(lambda h: cap_volume(h) - V0, x0=h0, fprime=dcap_volume, tol=1e-12)
    return np.sqrt(1 + h**2), h


def is_circular(nozzle) -> bool:
    return isinstance(nozzle, str) and nozzle == "circle"


def build_mesh(h, num_stacks, num_sectors, nozzle, collar_gap=0.05):
    '''
        Cap of the sphere of radius R = sqrt(1 + h^2) cut along a unit circle.

        nozzle = "circle": the cut lies on the plane z = 0 and the cut ring is
            the edge of a circular nozzle of radius 1 (fixed vertices).
        nozzle = array (M, 2): polygonal nozzle on z = 0. The cut is placed
            collar_gap below it and joined to it by a collar of triangles; the
            cut ring is free to move.

        Returns (mesh, center), where center is the centre of the sphere, the
        point the vertices move towards/away from.
    '''
    R = np.sqrt(1 + h**2)
    gap = 0.0 if is_circular(nozzle) else collar_gap
    center = point(0.0, 0.0, -h - gap)

    mesh = Mesh.cut_sphere(
        r=R,
        num_sectors=num_sectors,
        num_stacks=num_stacks,
        min_theta=np.arccos(h / R),
        z_center=center[2],
    )

    if is_circular(nozzle):
        mesh.use_cut_as_nozzle()
    else:
        if gap <= 0:
            raise ValueError("A polygonal nozzle needs collar_gap > 0.")
        mesh.attach_nozzle(np.asarray(nozzle, dtype=float), 0.0, 0.0)

    return mesh, center


def initial_mesh(droplet_Volume, num_stacks, num_sectors, nozzle,
                 collar_gap=0.05, QuadRule="strang3"):
    '''
        Initial mesh whose discrete volume is exactly droplet_Volume.

        The exact cap formula gives a first guess for h; the triangulated
        surface (and, for a polygonal nozzle, the collar) changes the volume
        a little, so h is then corrected by a scalar root search on the
        volume of the mesh itself.
    '''
    def V_mesh(h):
        mesh, center = build_mesh(h, num_stacks, num_sectors, nozzle, collar_gap)
        return volume_3D(np.zeros(len(mesh.free_ids)), mesh, center, QuadRule)

    _, h0 = sphere_from_volume(droplet_Volume)

    f = lambda h: V_mesh(h) - droplet_Volume                       # noqa: E731
    lo, hi, step = h0 - 0.25, h0 + 0.25, 0.25
    while f(lo) > 0:
        step *= 2
        lo -= step
    step = 0.25
    while f(hi) < 0:
        step *= 2
        hi += step
    h = optm.brentq(f, lo, hi, xtol=1e-14, rtol=1e-14)

    return build_mesh(h, num_stacks, num_sectors, nozzle, collar_gap)


# ---------------------------------------------------------------------------
# non-axisymmetric constraint: a flat wall
# ---------------------------------------------------------------------------

def wall_constraint(mesh: Mesh, center, w: float, normal=(1.0, 0.0, 0.0)):
    '''
        Keeps every vertex on the side n . P <= w of a flat wall.

        A free vertex moves as P_i = c + (1 + x_i) d_i, with d_i = P_i^0 - c,
        so n . P_i <= w is linear in x_i:  (n . d_i) x_i <= w - n . c - n . d_i.
        Only the vertices that move towards the wall (n . d_i > 0) give a row.
        The half-space is convex, so if the three vertices of a flat triangle
        satisfy it, the whole triangle does.

        Returns (scipy LinearConstraint, ids of the constrained free vertices).
    '''
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    c = np.asarray(center, dtype=float)

    ids = np.asarray(mesh.free_ids, dtype=int)
    d = np.asarray(mesh.points, dtype=float)[ids] - c
    nd = d @ n
    rows = np.flatnonzero(nd > 1e-12)

    A = np.zeros((len(rows), len(ids)))
    A[np.arange(len(rows)), rows] = nd[rows]
    ub = w - n @ c - nd[rows]

    return optm.LinearConstraint(A, -np.inf, ub), ids[rows]


# ---------------------------------------------------------------------------
# optimization
# ---------------------------------------------------------------------------

def find_droplet_optimization_3d(
    droplet_Volume: float,
    Bond: float,
    num_stacks: int,
    num_sectors: int,
    nozzle="circle",
    constraints: list | None = None,
    wall: float | None = None,
    QuadRule: str = "strang3",
    analytic_gradient: bool = True,
    collar_gap: float = 0.05,
    callback=None,
    verbose: bool = True,
    return_result: bool = False,
):
    '''
        Minimizes the energy of a triangulated droplet hanging from a nozzle,
        with fixed volume (SLSQP).

        nozzle: "circle" (radius 1, the cut ring is the nozzle edge) or an
            array (M, 2) with the vertices of a polygonal nozzle on z = 0.
        wall: if given, adds the flat wall x <= wall (see wall_constraint).
            wall = 1 with the circular nozzle puts the wall touching the nozzle.
        QuadRule: triangle quadrature rule ("strang3", "strang7" or "strang10").
        analytic_gradient: pass the exact gradients of the energy and of the
            volume to SLSQP (otherwise SciPy uses finite differences).
        callback: called by SLSQP after each iteration with the current x.
        return_result: also return the scipy OptimizeResult and the initial
            mesh/center, as (mesh, res, mesh0, center).
    '''
    # new list on every call: the caller's list is not modified and the
    # volume constraint does not accumulate between calls
    constraints = list(constraints or [])

    mesh, center = initial_mesh(droplet_Volume, num_stacks, num_sectors, nozzle,
                                collar_gap, QuadRule)

    # only the free vertices are variables; the nozzle is fixed
    N = len(mesh.free_ids)
    x0 = np.zeros(N)

    v = lambda x: volume_3D(x, mesh, center, QuadRule)                    # noqa: E731
    energy = lambda x: energy_3D(x, mesh, Bond, center, QuadRule)         # noqa: E731

    if analytic_gradient:
        jac = lambda x: grad_energy_3D(x, mesh, Bond, center, QuadRule)   # noqa: E731
        jac_v = lambda x: grad_volume_3D(x, mesh, center, QuadRule)[None, :]  # noqa: E731
        volume_constraint = optm.NonlinearConstraint(v, droplet_Volume, droplet_Volume, jac=jac_v)
    else:
        jac = None
        volume_constraint = optm.NonlinearConstraint(v, droplet_Volume, droplet_Volume)
    constraints.append(volume_constraint)

    if wall is not None:
        constraints.append(wall_constraint(mesh, center, wall)[0])

    lb = np.full(N, -1.0 + 1.0e-8)
    ub = np.full(N, np.inf)

    res = optm.minimize(
        energy,
        x0,
        jac=jac,
        method='SLSQP',
        constraints=constraints,
        bounds=optm.Bounds(lb, ub),
        callback=callback,
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

    final = deformed_mesh(res.x, mesh, center)
    if return_result:
        return final, res, mesh, center
    return final
