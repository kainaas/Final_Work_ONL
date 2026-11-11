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


def find_droplet_optimization(droplet_Volume:float, a: float, Bond: float, points_num: int,  constraints: List = []) -> Droplet:
    center = point(0.0, 0.0)
    arc = Arc(point(1.0, 0.0), point(0.0, -1.0), center, True, False)
    ctr = Contour(1.0)
    ctr.add_component(arc)
    ctr.discretize_n_lines(points_num-1)

    Lpoints0 = np.hstack([ctr.Lmatrix.copy()[0:2, :], np.reshape(ctr.Lmatrix.copy()[3:5, -1], (2,1))])
    N = Lpoints0.shape[1]
    x0 = np.zeros(N)
    
    v = lambda x: volume_droplet(x, Lpoints0, center)

    energy = lambda x: potential_energy(x, Lpoints0, center, Bond)

    volume_constraint = optm.NonlinearConstraint(v, droplet_Volume, droplet_Volume)

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
    z = np.flip(res_points[1,:])
    return Droplet(r, z, droplet_Volume, None, a, adimensional=True)





if __name__ == "__main__":
    gamma = 0.0728      #N/m
    density = 1000.0    #kg/m³
    a = 0.001
    Bond = a**2 * density * const.g / gamma
    # Bond = 0.0
    V_adim = 40.0
    droplet = find_droplet_optimization(V_adim, a, Bond, 100)
    
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect("equal")
    droplet.plot(ax)

    plt.show()