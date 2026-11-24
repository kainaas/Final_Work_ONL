from sys import path

path.append("./../src/2D")

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optm
from tabulate import tabulate

from droplet_optimization import *
from droplet import *

'''
Max volume droplet:
    A = 1.19721119513
    V* = 35.39486160126

Max pressure droplet:
    A* = 2.04614459832
    V = 2.17221828749
'''
V_max = 35.39486160126

gamma = 0.0728      #N/m
density = 1000.0    #kg/m³
a = 0.001
Bond = a**2 * density * const.g / gamma


what_to_do = "more_restriction" #"vals", "conv", "plot_lambda_V", "more_restriction"

if what_to_do == "vals" or what_to_do == "plot_lambda_V":
    
    Vlst: list
    if what_to_do == "vals":
        Vlst=[V_max/3.0, 2.0* V_max /3.0, V_max]
    else:
        Vlst = np.linspace(15, V_max, 20)
    lamb_shoot = []
    d_shoot = []

    lamb_optm = []
    d_optm = []

    for V_adim in Vlst:
        print("Initiating")

        A: float
        if V_adim <= 2.17221828749:
            A = shooting(V_adim, 0.0, 2.0, Bond, a)
        elif V_adim >35.3:
            A = 1.19721119514
        elif V_adim > 34.0:
            A = shooting(V_adim, 1.18, 1.20, Bond, a)
        elif V_adim > 20.0:
            A = shooting(V_adim, 1.2, 1.4, Bond, a)
        elif V_adim > 10.0:
            A = shooting(V_adim, 1.3, 1.6, Bond, a)
        elif V_adim > 5.0:
            A = shooting(V_adim, 1.5, 1.9, Bond, a)
        else:
            A = shooting(V_adim, 1.2, 1.9, Bond, a)


        ld = find_droplets(A, Bond, a, np.pi * 5.0, 0.001)
        
        drop_shooting: Droplet
        if V_adim > 35.3:
            drop_shooting = sorted(ld, key=lambda d: abs(d.V))[-1]
        else:
            if len(ld) >= 2:
                ld = ld[:2]
            drop_shooting = sorted(ld, key=lambda d: abs(d.V - V_adim))[0]
        lamb_shoot.append((A - Bond*drop_shooting.get_height())/np.pi)
        d_shoot.append(drop_shooting.get_height())
        print(drop_shooting.V)
        print("Shooting done\n\n")


        drop_optm, lamb = find_droplet_optimization(V_adim, a, Bond, 30, verbose=False)
        lamb_optm.append(lamb)
        d_optm.append(drop_optm.get_height())
        print("Optimization done\n\n")

        if what_to_do == "vals":
            fig, ax = plt.subplots(2,2)
            ax[0,0].set_aspect("equal")
            ax[0,1].set_aspect("equal")

            ax[1, 1].set_aspect("equal")


            drop_shooting.plot(ax[0,0], label="shooting")
            ax[0,0].set_title("Shooting")

            drop_optm.plot(ax[0,1], label="optimization")
            ax[0,1].set_title("optimization")

            drop_shooting.plot(ax[1,1], label="shooting")
            drop_optm.plot(ax[1,1], label="optimization", scatter=True)
            ax[1,1].legend()


    lamb_optm = np.array(lamb_optm)
    d_optm = np.array(d_optm)

    lamb_shoot = np.array(lamb_shoot)
    d_shoot = np.array(d_shoot)

    diff_lambda = np.abs(lamb_optm - lamb_shoot)
    diff_d = np.abs(d_optm - d_shoot)
    
    print(diff_d)
    print(diff_lambda)

    header = ["$\hat{V} / \hat{V}^*$", "\hat{d} EDO", "\hat{d} min", "\hat{\lambda} EDO", "\hat{\lambda} min"]

    lines = []
    for i in range(3):
        lines.append([(i+1)/3, d_shoot[i], d_optm[i], lamb_shoot[i], lamb_optm[i]])

    print(tabulate(lines, headers=header, tablefmt='latex'))

    if what_to_do == "plot_lambda_V":
        fig = plt.figure(figsize=(1000,1000, 'px'))
        ax = fig.add_subplot(111)
        ax.plot(Vlst, lamb_shoot, label="EDO")
        ax.scatter(Vlst, lamb_optm, label="Min")
        ax.set_title("Multiplicador de Lagrange em função do volume")
        ax.set_ylabel(r"$\hat{\lambda}$")
        ax.set_xlabel(r"$\hat{V}$")
        ax.legend()

elif what_to_do == "conv":
    V_adim = 2.0 * V_max /3.0
    A = shooting(V_adim, 1.2, 1.4, Bond, a)
    ld = find_droplets(A, Bond, a, np.pi * 5.0, 0.001)
    if len(ld) >= 2:
        ld = ld[:2]
    drop_shooting = sorted(ld, key=lambda d: abs(d.V - V_adim))[0]
    lamb_shoot = (A - Bond*drop_shooting.get_height())/np.pi
    d_shoot = drop_shooting.get_height()

    lamb_optm = []
    d_optm = []

    Nlst = [8, 16, 32, 64]    

    for N in Nlst:
        drop_optm, lamb = find_droplet_optimization(V_adim, a, Bond, N, verbose=False)
        lamb_optm.append(lamb)
        d_optm.append(drop_optm.get_height())
        print("Optimization done\n\n")
    
    diff_d = np.abs(d_optm - d_shoot)
    diff_lamb = np.abs(lamb_optm - lamb_shoot)

    header = ["N", r"$|\hat{d} - \hat{d}_{EDO}|$", r"$|\hat{\lambda} - \hat{\lambda}_{EDO}|$"]

    lines = []
    for i in range(4):
        lines.append([Nlst[i], diff_d[i], diff_lamb[i]])

    print(tabulate(lines, headers=header, tablefmt='latex'))





def find_droplet_optimization_restriction_r(
    droplet_Volume: float,
    a: float,
    Bond: float,
    points_num: int,
    r_max: float,
    constraints: Optional[list] = None,
    verbose: bool = True
):
    if constraints is None:
        constraints = []
    else:
        constraints = list(constraints)
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

    center = point(0.0, -h)
    arc = Arc(point(1.0, 0.0), point(0.0, -h - R), center, True, False)
    # center = point(0.0, 0.0)
    # arc = Arc(point(1.0, 0.0), point(0.0, -1.0), center, True, False)
    
    ctr = Contour(1.0)
    ctr.add_component(arc)
    ctr.discretize_n_lines(points_num-1)

    Lpoints0 = np.hstack([ctr.Lmatrix.copy()[0:2, :], np.reshape(ctr.Lmatrix.copy()[3:5, -1], (2,1))])
    N = Lpoints0.shape[1]
    x0 = np.zeros(N)
    
    v = lambda x: volume_droplet(x, Lpoints0, center)

    energy = lambda x: potential_energy(x, Lpoints0, center, Bond)

    volume_constraint = optm.NonlinearConstraint(v, droplet_Volume, droplet_Volume)

    A = np.diag(center[0] - Lpoints0[0, :])
    lbL = Lpoints0[0, :] - r_max
    ubL = np.full(N, np.inf)
    Lin_rest = optm.LinearConstraint(A, lbL, ubL)
    constraints.append(Lin_rest)

    constraints.append(volume_constraint)
    lb = np.full(N, -1.0 + 1.0e-8)
    ub = np.full(N,  np.inf)
    lb[0] = 0.0
    ub[0] = 0.0

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
    if verbose:
        print("Success:", res.success)
        print("Message:", res.message)
        print("Iterations:", res.nit)
        print("Energy:", res.fun)

        print("Target volume:", droplet_Volume)
        print("Computed volume:", volume_droplet(
            res.x, Lpoints0, center
        ))

        print("Constraint error:", abs(
            volume_droplet(res.x, Lpoints0, center)
            - droplet_Volume
        ))

        print("Gradient norm:", np.linalg.norm(res.jac))


    res_points, N = deform_droplet(res.x, Lpoints0, center)
    r = np.flip(res_points[0,:])
    z = np.flip(res_points[1,:]) - res_points[1,-1]


    def compute_lambda(y):

        grad_E = approx_derivative(energy, y).ravel()
        grad_V = approx_derivative(v, y).ravel()

        # Exclude the first variable, which is fixed.
        grad_E = grad_E[1:]
        grad_V = grad_V[1:]

        denominator = np.dot(grad_V, grad_V)

        if denominator <= const.EPSILON:
            raise RuntimeError("Volume gradient is too small.")

        return np.dot(grad_V, grad_E) / denominator

    return Droplet(r, z, droplet_Volume, None, a, adimensional=True), compute_lambda(res.x)




if what_to_do == "more_restriction":
    V_adim = 20.0
    drop_optm, lamb = find_droplet_optimization_restriction_r(V_adim, a, Bond, 30, 2.0, verbose=False)
    fig, ax = plt.subplots(1,1)
    ax.set_aspect("equal")
    drop_optm.plot(ax)



plt.show()
