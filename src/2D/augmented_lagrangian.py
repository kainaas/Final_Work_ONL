'''
    Augmented Lagrangian penalty method (Martins & Ning, "Engineering Design
    Optimization", Section 5.4.1, Algorithm 5.2), using scipy.optimize.minimize
    to solve each subproblem, and its application to the 2D (axisymmetric)
    pendant droplet of droplet_optimization.py.

        minimize    f(x)
        subject to  h(x) = 0         (equality constraints)
                    g(x) <= 0        (inequality constraints)
                    lb <= x <= ub    (bounds, passed to the subproblem solver)

    The multipliers follow the sign convention of the book:
        L(x, lambda, sigma) = f(x) + lambda . h(x) + sigma . g(x)

    Running this file (from src/2D) compares it with SLSQP and with the
    shooting method.
'''
import warnings

import numpy as np
import scipy.optimize as optm

from droplet import Droplet
from droplet_optimization import (deform_droplet, potential_energy,  # noqa: F401
                                  surface_droplet, volume_droplet,
                                  initial_profile, profile_energy, profile_volume)


def _vector(fun):
    '''
        Wraps a constraint function so that it always returns a 1D array
        (an empty one when there is no constraint of that kind).
    '''
    if fun is None:
        return lambda x: np.zeros(0)
    return lambda x: np.atleast_1d(np.asarray(fun(x), dtype=float))


def _matrix(jac, n: int):
    '''
        Wraps a constraint Jacobian so that it always returns an (m, n) array.
    '''
    if jac is None:
        return lambda x: np.zeros((0, n))
    return lambda x: np.asarray(jac(x), dtype=float).reshape(-1, n)


def _bounds_arrays(bounds, n: int):
    '''
        Lower and upper bounds as two arrays of size n (-inf / inf where free).
    '''
    if bounds is None:
        return np.full(n, -np.inf), np.full(n, np.inf)
    if isinstance(bounds, optm.Bounds):
        lb, ub = bounds.lb, bounds.ub
    else:
        pairs = [(-np.inf if lo is None else lo, np.inf if hi is None else hi)
                 for lo, hi in bounds]
        lb, ub = np.array(pairs, dtype=float).T
    return (np.broadcast_to(np.asarray(lb, dtype=float), (n,)).copy(),
            np.broadcast_to(np.asarray(ub, dtype=float), (n,)).copy())


def augmented_lagrangian(
    f,
    x0,
    h=None,
    g=None,
    jac=None,
    jac_h=None,
    jac_g=None,
    bounds=None,
    lambda0=None,
    sigma0=None,
    mu0: float = 1.0,
    rho: float = 2.0,
    mu_max: float = 1.0e12,
    tol: float = 1.0e-8,
    max_iter: int = 100,
    inner_method: str = "L-BFGS-B",
    inner_options: dict | None = None,
    fd_method: str = "2-point",
    verbose: bool = False,
) -> optm.OptimizeResult:
    '''
        Augmented Lagrangian penalty method (Alg. 5.2 of the book).

        Each outer iteration k minimizes, with scipy.optimize.minimize and
        starting from the previous solution, the augmented Lagrangian

            f^(x; lambda, sigma, mu) = f(x) + lambda . h(x) + (mu/2) |h(x)|^2
                                       + sigma . gbar(x) + (mu/2) |gbar(x)|^2

        where gbar_j(x) = max(g_j(x), -sigma_j / mu)        (Eqs. 5.46, 5.53, 5.54)

        and then updates

            lambda <- lambda + mu h(x)                           (Eq. 5.50)
            sigma  <- sigma + mu gbar(x)  ( = max(0, sigma + mu g(x)) )
            mu     <- rho mu

        It stops when max(|h(x)|, |gbar(x)|) <= tol. For the inequalities this
        measures feasibility and complementarity at the same time: gbar_j is
        g_j when the constraint is active and -sigma_j/mu (-> 0) when it is not.
        Since grad f^(x; lambda_k, sigma_k, mu_k) is the gradient of the
        Lagrangian with the updated multipliers, a converged subproblem plus
        this test gives an approximate KKT point.

        Safeguard (not part of Alg. 5.2): if a subproblem neither moves x nor
        reduces the violation, the violation is at the precision limit of the
        subproblem solutions (for instance, finite-difference gradients with a
        very small tol). The method then stops with success = False and keeps
        the multipliers of that subproblem: going on would only increase mu,
        and the updates mu h(x) would amplify the rounding noise of h into the
        multipliers.

        Parameters
        ----------
        f : callable f(x) -> float
        x0 : array (n,)
            Starting point.
        h, g : callable(x) -> float or array, optional
            Equality (h = 0) and inequality (g <= 0) constraints.
        jac, jac_h, jac_g : callable, optional
            Gradient of f (n,) and Jacobians of h and g (m, n). If all the ones
            needed are given, the gradient of f^ is assembled analytically
            (Eq. 5.47); otherwise the subproblem solver uses finite differences
            (fd_method).
        bounds : scipy.optimize.Bounds or sequence of (min, max), optional
            Bounds on x. They are not penalized: the subproblem solver
            (L-BFGS-B by default) enforces them.
        lambda0, sigma0 : array, optional
            Initial multipliers (zero by default, as in the book).
        mu0 : float
            Initial penalty parameter (> 0).
        rho : float
            Penalty increase factor (> 1).
        mu_max : float
            Cap on the penalty parameter (safeguard, not part of Alg. 5.2).
        tol : float
            Tolerance on the constraint violation.
        max_iter : int
            Maximum number of outer iterations.
        inner_method, inner_options : str, dict
            Method and options of scipy.optimize.minimize for the subproblems.

        Returns
        -------
        scipy.optimize.OptimizeResult with
            x, fun : solution and f(x)
            lam, sigma : Lagrange multipliers of h and g
            mu : last penalty parameter
            maxcv : constraint violation at x
            kkt : max |projected gradient of the Lagrangian| at x (only with
                analytic derivatives, None otherwise)
            nit : number of outer iterations
            nfev : function evaluations of all the subproblems
            success, message
            history : one dict per outer iteration k with the values used in
                the subproblem k (mu, lam, sigma) and its solution
                (x, f, violation, kkt, inner_nit, inner_success)
    '''
    if mu0 <= 0.0:
        raise ValueError("mu0 must be positive.")
    if rho <= 1.0:
        raise ValueError("rho must be greater than 1.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    x = np.asarray(x0, dtype=float).ravel().copy()
    n = x.size
    lb, ub = _bounds_arrays(bounds, n)
    h_vec, g_vec = _vector(h), _vector(g)
    Jh, Jg = _matrix(jac_h, n), _matrix(jac_g, n)
    m_h, m_g = h_vec(x).size, g_vec(x).size

    lam = np.zeros(m_h) if lambda0 is None else np.asarray(lambda0, dtype=float).reshape(m_h).copy()
    sig = np.zeros(m_g) if sigma0 is None else np.asarray(sigma0, dtype=float).reshape(m_g).copy()
    if np.any(sig < 0.0):
        raise ValueError("The inequality multipliers sigma0 must be nonnegative.")
    mu = float(mu0)

    analytic = (jac is not None
                and (m_h == 0 or jac_h is not None)
                and (m_g == 0 or jac_g is not None))

    options = {}
    if inner_method.upper() == "L-BFGS-B":
        # with exact gradients the subproblem only stops on gtol (or when the
        # line search can no longer progress); with finite differences the
        # relative reduction test (ftol) avoids iterating on gradient noise
        options = {"maxiter": 10000, "maxfun": 100000, "gtol": 1e-10,
                   "ftol": 0.0 if analytic else 1e-15}
    options.update(inner_options or {})

    def penalized(x, lam, sig, mu):
        hx, gx = h_vec(x), g_vec(x)
        gbar = np.maximum(gx, -sig / mu)                                # Eq. 5.54
        value = (f(x) + lam @ hx + 0.5 * mu * (hx @ hx)
                 + sig @ gbar + 0.5 * mu * (gbar @ gbar))               # Eq. 5.53
        return float(value), hx, gx

    def penalized_and_grad(x, lam, sig, mu):
        value, hx, gx = penalized(x, lam, sig, mu)
        grad = (np.asarray(jac(x), dtype=float)
                + Jh(x).T @ (lam + mu * hx)                             # Eq. 5.47
                + Jg(x).T @ np.maximum(0.0, sig + mu * gx))
        return value, grad

    history = []
    nfev = 0
    violation = np.inf
    success = stalled = False
    for k in range(max_iter):
        x_prev, violation_prev = x, violation
        # minimize f^(x; lambda_k, sigma_k, mu_k) starting from x_k
        if analytic:
            inner = optm.minimize(lambda x: penalized_and_grad(x, lam, sig, mu), x,
                                  jac=True, method=inner_method, bounds=bounds,
                                  options=options)
        else:
            inner = optm.minimize(lambda x: penalized(x, lam, sig, mu)[0], x,
                                  jac=fd_method, method=inner_method, bounds=bounds,
                                  options=options)
        x = np.asarray(inner.x, dtype=float)
        nfev += inner.nfev

        hx, gx = h_vec(x), g_vec(x)
        gbar = np.maximum(gx, -sig / mu)
        violation = float(max(np.max(np.abs(hx), initial=0.0),
                              np.max(np.abs(gbar), initial=0.0)))
        step = float(np.max(np.abs(x - x_prev), initial=0.0))
        stalled = (violation >= violation_prev
                   and step <= 1e-8 * (1.0 + float(np.max(np.abs(x), initial=0.0))))

        fx = float(f(x))
        record = {"k": k, "mu": mu, "lam": lam.copy(), "sigma": sig.copy(), "x": x.copy(),
                  "f": fx, "violation": violation, "inner_nit": inner.nit,
                  "inner_success": inner.success}

        if not stalled:
            # update the Lagrange multipliers (Eq. 5.50)
            lam = lam + mu * hx
            sig = sig + mu * gbar                 # = max(0, sigma + mu g(x))

        kkt = None
        if analytic:
            # gradient of the Lagrangian with the current multipliers; projected
            # on the bounds, it is the first-order optimality measure
            grad_L = np.asarray(jac(x), dtype=float) + Jh(x).T @ lam + Jg(x).T @ sig
            kkt = float(np.max(np.abs(x - np.clip(x - grad_L, lb, ub)), initial=0.0))
        record["kkt"] = kkt
        history.append(record)

        if verbose:
            info = f"  kkt={kkt:.3e}" if kkt is not None else ""
            if m_h + m_g <= 3:
                info += f"  lambda={lam}" + (f"  sigma={sig}" if m_g else "")
            print(f"k={k:3d}  mu={mu:.3e}  f={fx:.12g}  violation={violation:.3e}"
                  f"  inner it={inner.nit}{info}")

        if violation <= tol:
            success = True
            break
        if stalled:
            break
        mu = min(rho * mu, mu_max)                # increase the penalty parameter

    if success:
        message = f"Constraint violation {violation:.3e} <= tol after {len(history)} outer iterations."
    elif stalled:
        message = (f"Stalled with constraint violation {violation:.3e} > tol: the subproblem no "
                   "longer improves (precision limit). Returning the multipliers of the last subproblem.")
    else:
        message = f"Maximum number of outer iterations reached (violation {violation:.3e})."

    return optm.OptimizeResult(x=x, fun=float(f(x)), lam=lam, sigma=sig, mu=mu,
                               maxcv=violation, kkt=history[-1]["kkt"],
                               nit=len(history), nfev=nfev, success=success,
                               message=message, history=history)


# ---------------------------------------------------------------------------
# 2D (axisymmetric) pendant droplet
# ---------------------------------------------------------------------------

def find_droplet_augmented_lagrangian(
    droplet_Volume: float,
    a: float,
    Bond: float,
    points_num: int,
    mu0: float = 1000.0,
    rho: float = 2.0,
    lambda0: float = 0.0,
    tol: float = 1.0e-10,
    max_iter: int = 100,
    analytic_gradient: bool = True,
    inner_options: dict | None = None,
    verbose: bool = True,
    return_result: bool = False,
):
    '''
        Same problem as find_droplet_optimization (minimum energy with fixed
        volume, the point on the nozzle edge fixed, x_i > -1), solved with the
        augmented Lagrangian method instead of SLSQP.

        The volume constraint is scaled to order 1 (Tip 5.3 of the book):
            h(y) = V(y) / droplet_Volume - 1 = 0

        mu0: the default is large on purpose. In the first subproblem the
        relative volume error is about |lambda* - lambda0| V0 / mu0 (lambda*
        is the multiplier below). With a small mu0 the drop first shrinks a
        lot, and near the maximum volume V* (where the pendant equilibrium is
        only a local minimum) the method can then end on a pinched, detached
        profile of lower energy, with points collapsed on the bound x = -1; a
        warning is issued when that happens. With mu0 = 1000 the first
        subproblem already stays within about 1% of the volume.

        lambda0: initial estimate of the multiplier, in the convention of the
        returned lambda (for instance, the result for a nearby volume).

        analytic_gradient: if True, uses profile_energy / profile_volume and
        their exact gradients; if False, uses potential_energy / volume_droplet
        (the functions used with SLSQP) with finite differences.

        Returns (droplet, lambda), where lambda is the Lagrange multiplier in
        the convention of find_droplet_optimization (grad E = lambda grad V).
        Here it comes directly from the method, with no extra computation.
        With return_result=True, also returns the OptimizeResult of
        augmented_lagrangian, which holds the iteration history.
    '''
    Lpoints0, center = initial_profile(droplet_Volume, points_num)
    D = Lpoints0 - center.reshape(2, 1)       # d points[:, i] / d x_i
    N = Lpoints0.shape[1]

    # x_0 = 0 keeps the point on the nozzle edge fixed; y = x[1:] are the variables
    def full(y):
        return np.concatenate([[0.0], y])

    if analytic_gradient:
        def points(y):
            return Lpoints0 + full(y) * D

        def f(y):
            return profile_energy(points(y), Bond)[0]

        def jac(y):
            return np.sum(profile_energy(points(y), Bond)[1] * D, axis=0)[1:]

        def h(y):
            return profile_volume(points(y))[0] / droplet_Volume - 1.0

        def jac_h(y):
            return np.sum(profile_volume(points(y))[1] * D, axis=0)[None, 1:] / droplet_Volume
    else:
        def f(y):
            return potential_energy(full(y), Lpoints0, center, Bond)

        def h(y):
            return volume_droplet(full(y), Lpoints0, center) / droplet_Volume - 1.0

        jac = jac_h = None

    lower = -1.0 + 1.0e-8
    bounds = optm.Bounds(np.full(N - 1, lower), np.full(N - 1, np.inf))
    # book convention: L = E + lam (V/V0 - 1)  =>  grad E = (-lam / V0) grad V
    res = augmented_lagrangian(f, np.zeros(N - 1), h=h, jac=jac, jac_h=jac_h,
                               bounds=bounds, lambda0=[-lambda0 * droplet_Volume],
                               mu0=mu0, rho=rho, tol=tol, max_iter=max_iter,
                               inner_options=inner_options, verbose=verbose)
    lamb = -float(res.lam[0]) / droplet_Volume

    collapsed = np.count_nonzero(res.x <= lower + 1.0e-6)
    if collapsed > 0:
        warnings.warn(
            f"{collapsed} profile points reached the bound x = -1 (they collapsed on "
            "the center of the initial cap): this is probably a pinched, detached "
            "drop and not the pendant equilibrium. Try a larger mu0 or a lambda0 "
            "estimate.", RuntimeWarning, stacklevel=2)

    x = full(res.x)
    res_points, _ = deform_droplet(x, Lpoints0, center)
    r = np.flip(res_points[0, :])
    z = np.flip(res_points[1, :]) - res_points[1, -1]

    if verbose:
        V = volume_droplet(x, Lpoints0, center)
        print("Success:", res.success)
        print("Message:", res.message)
        print("Outer iterations:", res.nit, " Function evaluations:", res.nfev)
        print("Energy:", res.fun)
        print("Target volume:", droplet_Volume)
        print("Computed volume:", V)
        print("Constraint error:", abs(V - droplet_Volume))
        if res.kkt is not None:
            print("Max |projected gradient of L|:", res.kkt)
        print("Lagrange multiplier:", lamb)

    droplet = Droplet(r, z, droplet_Volume, surface_droplet(res_points), a, adimensional=True)
    if return_result:
        return droplet, lamb, res
    return droplet, lamb


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    import const
    from droplet import find_droplets, shooting
    from droplet_optimization import find_droplet_optimization

    gamma = 0.0728      #N/m
    density = 1000.0    #kg/m³
    a = 0.001
    Bond = a**2 * density * const.g / gamma

    V_max = 35.39486160126
    V_adim = V_max / 3.0
    points_num = 30

    t = time.perf_counter()
    drop_al, lamb_al = find_droplet_augmented_lagrangian(V_adim, a, Bond, points_num)
    t_al = time.perf_counter() - t

    t = time.perf_counter()
    drop_sqp, lamb_sqp = find_droplet_optimization(V_adim, a, Bond, points_num, verbose=False)
    t_sqp = time.perf_counter() - t

    # reference: shooting method (the bracket [1.3, 1.6] is the one used in
    # run/comparison.py for 10 < V <= 20)
    A = shooting(V_adim, 1.3, 1.6, Bond, a)
    ld = find_droplets(A, Bond, a, np.pi * 5.0, 0.001)[:2]
    drop_ode = min(ld, key=lambda d: abs(d.V - V_adim))
    lamb_ode = (A - Bond * drop_ode.get_height()) / np.pi

    print()
    print(f"{'method':<22}{'height d':>14}{'lambda':>14}{'time (s)':>10}")
    print(f"{'shooting (ODE)':<22}{drop_ode.get_height():>14.8f}{lamb_ode:>14.8f}{'':>10}")
    print(f"{'SLSQP':<22}{drop_sqp.get_height():>14.8f}{lamb_sqp:>14.8f}{t_sqp:>10.2f}")
    print(f"{'augmented Lagrangian':<22}{drop_al.get_height():>14.8f}{lamb_al:>14.8f}{t_al:>10.2f}")

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect("equal")
    drop_ode.plot(ax, label="shooting (ODE)")
    drop_al.plot(ax, label="augmented Lagrangian")
    ax.legend()
    plt.show()
