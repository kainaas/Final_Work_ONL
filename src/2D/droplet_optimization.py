import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optm

import const
from droplet import *
from quadrature import Gauss_Legendre
from objects import Arc, Contour, param, point


def deform_droplet(x: np.array, Lpoints0: np.array, center: np.array):
    x = np.asarray(x, dtype=float).reshape(1, -1)
    N = x.shape[1]

    return Lpoints0 + np.vstack([x, x]) * (Lpoints0 - np.asarray(center).reshape(2, 1)), N

def potential_energy(x: np.array, Lpoints0: np.array, center:np.array, Bond: float) -> float:
    points, N = deform_droplet(x, Lpoints0, center)
    Lmatrix = np.vstack([points[:, :-1], np.ones((1,N-1)), points[:,1:], np.ones((1,N-1))])
    d = np.sqrt((Lmatrix[0,:] - Lmatrix[3,:])**2 + (Lmatrix[1,:] - Lmatrix[4,:])**2)
    assert np.all(d > 0.0)
    normals = np.vstack([(Lmatrix[1,:] - Lmatrix[4,:]) / d, (Lmatrix[3,:] - Lmatrix[0,:]) / d])
    def f(xi:float):
        v = param(Lmatrix, xi)
        return np.linalg.norm((Lmatrix[0:2, :] - Lmatrix[3:5, :])/2,axis=0) * v[0,:] * (2.0 + Bond * v[1,:]**2 * normals[1,:])
    return Gauss_Legendre.integrate_vector(f)


def surface_droplet(points: np.array) -> float:
    '''
        Area of the surface of revolution of the polygonal profile points (2, N),
        rows (r, z). Each segment generates a conical frustum of area
        2 pi * mean radius * length, so the result is exact for the polygon.
    '''
    r_mean = (points[0, :-1] + points[0, 1:]) / 2.0
    lengths = np.linalg.norm(np.diff(points, axis=1), axis=0)
    return float(2.0 * np.pi * np.sum(r_mean * lengths))


def volume_droplet(x: np.array, Lpoints0: np.array, center:np.array) -> float:
    points, N = deform_droplet(x, Lpoints0, center)
    Lmatrix = np.vstack([points[:, :-1], np.ones((1,N-1)), points[:,1:], np.ones((1,N-1))])
    d = np.sqrt((Lmatrix[0,:] - Lmatrix[3,:])**2 + (Lmatrix[1,:] - Lmatrix[4,:])**2)
    assert np.all(d > 0.0)
    normals = np.vstack([(Lmatrix[1,:] - Lmatrix[4,:]) / d, (Lmatrix[3,:] - Lmatrix[0,:]) / d])
    def f(xi:float):
        v = param(Lmatrix, xi)
        return np.linalg.norm((Lmatrix[0:2, :] - Lmatrix[3:5, :])/2,axis=0) * v[0,:] * (v[0,:] * normals[0, :] + v[1,:] * normals[1,:])
    return (2.0/3.0) * np.pi * Gauss_Legendre.integrate_vector(f)




from typing import Optional


def profile_energy(points: np.ndarray, Bond: float):
    '''
        Energy of the polygonal profile points (2, N), rows (r, z), with the
        same normalization as potential_energy (energy / pi):

            sum over segments of  d (r1 + r2) + (Bond/12) (r2 - r1) Q
            Q = r1 (3 z1^2 + 2 z1 z2 + z2^2) + r2 (z1^2 + 2 z1 z2 + 3 z2^2)

        where d is the segment length. This is the exact integral of the
        integrand of potential_energy on each segment (the 3-point
        Gauss-Legendre rule is exact for it), so both agree up to rounding.

        Returns (E, dE/dpoints), the gradient with shape (2, N).
    '''
    r1, z1, r2, z2 = points[0, :-1], points[1, :-1], points[0, 1:], points[1, 1:]
    dr, dz = r2 - r1, z2 - z1
    d = np.hypot(dr, dz)
    s = r1 + r2
    Q = r1 * (3*z1**2 + 2*z1*z2 + z2**2) + r2 * (z1**2 + 2*z1*z2 + 3*z2**2)
    c = Bond / 12.0
    E = np.sum(d * s + c * dr * Q)

    grad = np.zeros(points.shape)
    grad[0, :-1] += -dr / d * s + d + c * (-Q + dr * (3*z1**2 + 2*z1*z2 + z2**2))
    grad[0, 1:]  +=  dr / d * s + d + c * ( Q + dr * (z1**2 + 2*z1*z2 + 3*z2**2))
    grad[1, :-1] += -dz / d * s + c * dr * (r1 * (6*z1 + 2*z2) + r2 * (2*z1 + 2*z2))
    grad[1, 1:]  +=  dz / d * s + c * dr * (r1 * (2*z1 + 2*z2) + r2 * (2*z1 + 6*z2))
    return float(E), grad


def profile_volume(points: np.ndarray):
    '''
        Volume between the surface of revolution of the polygonal profile
        points (2, N) and the nozzle plane z = 0 (same value as volume_droplet):

            (2 pi / 3) sum over segments of
                (z1 - z2) (r1^2 + r1 r2 + r2^2) / 3
              + (r2 - r1) (2 r1 z1 + r1 z2 + r2 z1 + 2 r2 z2) / 6

        Returns (V, dV/dpoints), the gradient with shape (2, N).
    '''
    r1, z1, r2, z2 = points[0, :-1], points[1, :-1], points[0, 1:], points[1, 1:]
    dr, dzm = r2 - r1, z1 - z2
    S2 = r1**2 + r1*r2 + r2**2
    M = 2*r1*z1 + r1*z2 + r2*z1 + 2*r2*z2
    c = 2.0 * np.pi / 3.0
    V = c * np.sum(dzm * S2 / 3.0 + dr * M / 6.0)

    grad = np.zeros(points.shape)
    grad[0, :-1] += c * (dzm * (2*r1 + r2) / 3.0 + (-M + dr * (2*z1 + z2)) / 6.0)
    grad[0, 1:]  += c * (dzm * (r1 + 2*r2) / 3.0 + ( M + dr * (z1 + 2*z2)) / 6.0)
    grad[1, :-1] += c * ( S2 / 3.0 + dr * (2*r1 + r2) / 6.0)
    grad[1, 1:]  += c * (-S2 / 3.0 + dr * (r1 + 2*r2) / 6.0)
    return float(V), grad


def initial_profile(droplet_Volume: float, points_num: int):
    '''
        Starting geometry of find_droplet_optimization and of
        find_droplet_augmented_lagrangian: a spherical cap hanging from the
        nozzle edge (r, z) = (1, 0), discretized in points_num points from the
        nozzle edge to the axis.

        The exact cap formula gives a first guess for h (the centre is at
        z = -h); the polygon inscribed in the arc holds a little less water, so
        h is then corrected by a scalar root search on the volume of the
        polygon itself (as initial_mesh does in 3D). The optimization therefore
        starts feasible.

        Returns (Lpoints0 (2, points_num), center (2,)).
    '''
    def cap(h):
        R = np.sqrt(1 + h**2)
        center = point(0.0, -h)
        arc = Arc(point(1.0, 0.0), point(0.0, -h - R), center, True, False)
        ctr = Contour(1.0)
        ctr.add_component(arc)
        ctr.discretize_n_lines(points_num - 1)
        return np.hstack([ctr.Lmatrix[0:2, :], ctr.Lmatrix[3:5, -1:]]), center

    def volume(h):
        return np.pi * (h + 2/3 * (h**3 + (1 + h**2)**1.5))

    def dvolume(h):
        return np.pi * (np.sqrt(1 + h**2) + h)**2

    h0 = optm.newton(lambda h: volume(h) - droplet_Volume, x0=0.0, fprime=dvolume, tol=1e-12)

    f = lambda h: profile_volume(cap(h)[0])[0] - droplet_Volume      # noqa: E731
    lo, hi, step = h0 - 0.25, h0 + 0.25, 0.25
    while f(lo) > 0:
        step *= 2
        lo -= step
    step = 0.25
    while f(hi) < 0:
        step *= 2
        hi += step
    h = optm.brentq(f, lo, hi, xtol=1e-14, rtol=1e-14)
    return cap(h)


# objective scale of SLSQP (see find_droplet_optimization)
OBJECTIVE_SCALE = 0.01


def find_droplet_optimization(
    droplet_Volume: float,
    a: float,
    Bond: float,
    points_num: int,
    constraints: Optional[list] = None,
    verbose: bool = True,
    callback=None,
    return_result: bool = False,
):
    '''
        Minimum energy profile with fixed volume (SLSQP), starting from the
        spherical cap of initial_profile, with the point on the nozzle edge
        fixed (x_0 = 0) and x_i > -1.

        The exact gradients of the energy and of the volume (profile_energy,
        profile_volume) are passed to SLSQP. The energy is multiplied by
        OBJECTIVE_SCALE: SLSQP starts with B_0 = I, so its first steps have the
        size of the energy gradient, and without the scale the first step
        leaves the volume constraint far behind (V = 20 jumps to 35 before
        coming back). The minimizer does not change; the multiplier is divided
        back by the scale.

        callback: if given, called after each SLSQP iteration with the
        profile points (2, N), rows (r, z), from the nozzle to the apex.
        return_result: also return the scipy OptimizeResult (res.x holds
        x_1 ... x_{N-1}; x_0 = 0 is not a variable).

        Returns (droplet, lambda), where lambda is the Lagrange multiplier of
        the volume constraint, grad E = lambda grad V (E divided by pi).
    '''
    constraints = list(constraints or [])

    Lpoints0, center = initial_profile(droplet_Volume, points_num)
    D = Lpoints0 - center.reshape(2, 1)       # d points[:, i] / d x_i
    N = Lpoints0.shape[1]
    s = OBJECTIVE_SCALE

    # x_0 = 0 keeps the point on the nozzle edge fixed; y = x[1:] are the variables
    def full(y):
        return np.concatenate([[0.0], y])

    def points(y):
        return Lpoints0 + full(y) * D

    def energy(y):
        return s * profile_energy(points(y), Bond)[0]

    def jac(y):
        return s * np.sum(profile_energy(points(y), Bond)[1] * D, axis=0)[1:]

    def v(y):
        return profile_volume(points(y))[0]

    def jac_v(y):
        return np.sum(profile_volume(points(y))[1] * D, axis=0)[None, 1:]

    constraints.insert(0, optm.NonlinearConstraint(v, droplet_Volume, droplet_Volume, jac=jac_v))

    res = optm.minimize(
        energy,
        np.zeros(N - 1),
        jac=jac,
        method='SLSQP',
        constraints=constraints,
        bounds=optm.Bounds(np.full(N - 1, -1.0 + 1.0e-8), np.full(N - 1, np.inf)),
        callback=None if callback is None else (lambda yk: callback(points(yk))),
        options={
            'maxiter': 10000,
            'ftol': 1e-12 * s,
            'disp': verbose
        }
    )

    # SciPy: grad f = sum_j multipliers_j grad c_j; the volume constraint is the first one
    lam = float(np.ravel(res.multipliers)[0]) / s

    if verbose:
        print("Success:", res.success)
        print("Message:", res.message)
        print("Iterations:", res.nit)
        print("Energy:", res.fun / s)
        print("Target volume:", droplet_Volume)
        print("Computed volume:", v(res.x))
        print("Constraint error:", abs(v(res.x) - droplet_Volume))
        # gradient of the Lagrangian: it vanishes at the solution
        # (the gradient of the energy alone does not, because of the constraint)
        print("Lagrangian gradient norm:", np.linalg.norm(jac(res.x) / s - lam * jac_v(res.x)[0]))

    res_points = points(res.x)
    r = np.flip(res_points[0, :])
    z = np.flip(res_points[1, :]) - res_points[1, -1]

    S = surface_droplet(res_points)
    drop = Droplet(r, z, droplet_Volume, S, a, adimensional=True)
    if return_result:
        return drop, lam, res
    return drop, lam


if __name__ == "__main__":
    gamma = 0.0728      #N/m
    density = 1000.0    #kg/m³
    a = 0.001
    Bond = a**2 * density * const.g / gamma
    # Bond = 0.0
    V_adim = 40.0
    droplet, res = find_droplet_optimization(V_adim, a, Bond, 100)
    
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect("equal")
    droplet.plot(ax)

    plt.show()