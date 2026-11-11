from solve_PVI import *
import const
from objects import Circle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools


class Droplet:
    '''
        Parameters:
        -----------
        r: x coordinates of the droplet
        z: z coordinates of the droplet
        V: volume of the 3D droplet 
        S: surface area of the droplet
        a: is the adimensionalization factor. For pendant droplets is the radius of the nozzle. For sessile droplets is the capillary length
        d_type: "pendant" or "sessile" droplet
        adimensional: tells if the droplet is adimensionalized 
        '''
    def __init__(self, 
        r:np.array, 
        z:np.array, 
        V:float, 
        S:float,
        a: float,
        d_type: str = "pendant",
        adimensional:bool = True
    ):
        self.r = r
        self.z = z
        self.V = V
        self.S = S
        self.a = a
        self.d_type = d_type
        self.adimensional = adimensional
        if d_type not in ("pendant", "sessile"):
            raise ValueError(
                "d_type must be 'pendant' or 'sessile'."
            )

    def get_plot_vecs(self):
        x_plot = np.hstack([np.flip(-self.r), self.r])
        z_plot = np.hstack([np.flip(self.z), self.z])
        if self.d_type == "pendant":
            z_plot = z_plot - self.get_height()
        else:
            z_plot = self.get_height() - z_plot
        return x_plot, z_plot

    def plot(self, ax:plt.Axes, label: str|None = None):
        x_plot, z_plot = self.get_plot_vecs()
        if label is None:
            ax.plot(x_plot, z_plot)
        else:
            ax.plot(x_plot, z_plot, label=label)

    def dimensionalize(self):
        if self.adimensional:
            self.r = self.a * self.r
            self.z = self.a * self.z
            self.V = self.V * self.a**3
            self.S = self.S * self.a**2
            self.adimensional = False
        else:
            print("Already dimensionalized")

    def adimensionalize(self):
        if not self.adimensional:
            self.r = self.r / self.a
            self.z = self.z / self.a
            self.V = self.V / self.a**3
            self.S = self.S / self.a**2
            self.adimensional = True
        else:
            print("Already adimensionalized")
    
    def change_a(self, new_a:float):
        if new_a <= 0:
            if self.d_type == "pendant":
                raise ValueError("The nozzle radius must be positive.")
            else:
                raise ValueError("The characteristic length must be positive.")
        if not self.adimensional:
            self.adimensionalize()
            self.a = new_a
            self.dimensionalize()
        else:
            self.a = new_a


    def get_height(self) -> float:
        return self.z[-1]



'''
u = r   =   u[0]
    z       u[1]
    theta   u[2]
    V       u[3]
    S       u[4]
'''

'''
pendant:
f(u,s) = cos(theta)                                     =   cos(u[2])
         sin(theta)                                         sin(u[2])
         (p0 - density * g  * z) / gamma - sin(theta)/r     (p0 - density * g * u[1]) / gamma - sin(u[2])/u[0]
         pi * r * r * sin(theta)                            pi * u[0] * u[0] * sin(u[2])
         2 * pi * r                                         2 * pi * u[0]
sessile:
f(u,s) = cos(theta)                                     =   cos(u[2])
         sin(theta)                                         sin(u[2])
         (p0 + density * g  * z) / gamma - sin(theta)/r     (p0 - density * g * u[1]) / gamma - sin(u[2])/u[0]
         pi * r * r * sin(theta)                            pi * u[0] * u[0] * sin(u[2])
         2 * pi * r                                         2 * pi * u[0]

'''

def f_aux_droplet(
    u: np.array,    #Last vector
    s: float,       #
    gamma: float,   #Superficial tension
    p0: float,      #Initial pressure
    density: float  #Liquid density
) -> np.array:
    udiv: float
    if np.abs(u[0]) < const.EPSILON: udiv = const.EPSILON
    else: udiv = u[0]
    return np.array([
        np.cos(u[2]),
        np.sin(u[2]),
        (p0 - density * const.g * u[1]) / gamma - np.sin(u[2])/udiv,
        np.pi * u[0] * u[0] * np.sin(u[2]),
        2 * np.pi * u[0]
    ])

'''
A = a * p0 / gamma
B = rho * g * a² / gamma
'''

def f_aux_droplet_adimensionalized(
    u: np.array,    #Last vector
    s: float,       #
    A: float,
    B: float,
    d_type: str = "pendant"
) -> np.array:
    if d_type == "pendant":
        return np.array([
            np.cos(u[2]),
            np.sin(u[2]),
            A - B * u[1] - np.sin(u[2])/u[0],
            np.pi * u[0] * u[0] * np.sin(u[2]),
            2 * np.pi * u[0]
        ])
    else:
        return np.array([
            np.cos(u[2]),
            np.sin(u[2]),
            A + B * u[1] - np.sin(u[2])/u[0],
            np.pi * u[0] * u[0] * np.sin(u[2]),
            2 * np.pi * u[0]
        ])

def initialize_droplet(
    A: float,
    B: float,
    eps: float = 1.0e-7,
    d_type: str = "pendant"
) -> np.array:
    if d_type == "pendant":
        return np.array([
            eps - (A*A/24.0)* eps**3, #r
            (A/4.0)*eps*eps, #z
            (A/2.0)*eps - (A*B/16.0)* eps**3, #theta
            (np.pi * A /8.0)*eps**4, #V
            np.pi*eps*eps
        ])
    else:
        return np.array([
            eps - (A*A/24.0)* eps**3, #r
            (A/4.0)*eps*eps, #z
            (A/2.0)*eps + (A*B/16.0)* eps**3, #theta
            (np.pi * A /8.0)*eps**4, #V
            np.pi*eps*eps
        ])

def find_args_intersection(result: Result, stop_val: float, d_type: str = "pendant") -> list:
    '''
    Find the arguments of the result such that it has the same radius as a.
    The returned argument i is such that u[j, i] <= stop_val <= u[j,i+1], j= 0 or 2
    '''
    tmp: np.ndarray
    if d_type == "pendant":
        tmp = np.vstack([result.u[0,:-1], result.u[0,1:]])
    else: #Sessile drop
        tmp = np.vstack([result.u[2,:-1], result.u[2,1:]])
    args = np.argwhere(((tmp[0,:] <= stop_val) & (tmp[1,:] > stop_val)) | ((tmp[0,:] >= stop_val) & (tmp[1,:] < stop_val)) | (tmp[0,:] == stop_val) & (tmp[1,:] == stop_val))
    return list(itertools.chain.from_iterable(args))


def find_droplets(
    A: float,
    B: float,
    a: float,
    max_s: float,
    ds: float,
    stop_val: float = 1.0,
    d_type: str = "pendant"
) -> list[Droplet]:
    f = lambda u,s: f_aux_droplet_adimensionalized(u,s, A, B, d_type)
    eps = 1.0e-7
    u0 = initialize_droplet(A, B, eps, d_type)
    result = RK4(eps, max_s, ds, u0, f)
    args = find_args_intersection(result, stop_val, d_type)
    
    droplets = []
    if len(args) == 0:
        return droplets

    if d_type == "sessile":
        args = [args[0]]
    for arg in args:
        r = result.u[0,:arg]
        z = result.u[1,:arg]
        V = result.u[3, arg]
        S = result.u[4, arg]
        if arg < len(result.u[0,:]) - 1:
            u_arg = result.u[:, arg]
            u_rarg = result.u[:, arg+1]
            if np.abs(u_arg[0] - u_rarg[0]) < const.EPSILON:
                d = Droplet(r, z, V, S, a, d_type)
                droplets.append(d)
            else:
                stop_idx = 0 if d_type == "pendant" else 2
                xi = (stop_val - u_rarg[stop_idx]) / (u_arg[stop_idx] - u_rarg[stop_idx])
                u_interpol = xi * u_arg + (1.0 - xi) * u_rarg
                u_interpol = xi * u_arg + (1.0 - xi) * u_rarg
                r = np.hstack([r, u_interpol[0]])
                z = np.hstack([z, u_interpol[1]])
                V = u_interpol[3]
                S = u_interpol[4]
                d = Droplet(r, z, V, S, a, d_type)
                droplets.append(d)
        else: print("Run with bigger max_s. Intersection on the edge of the solution vector")
    return droplets
        
    
def max_V_droplet(
    A: float,
    B: float,
    a: float,
    max_s: float,
    ds: float
) -> Droplet:
    ld = find_droplets(A, B, a, max_s, ds)
    max_V_droplet: Droplet = None
    for d in ld:
        if (max_V_droplet is None) or max_V_droplet.V < d.V:
            max_V_droplet = d
    return max_V_droplet


def min_V_droplet(
    A: float,
    B: float,
    a: float,
    max_s: float,
    ds: float
) -> Droplet:
    ld = find_droplets(A, B, a, max_s, ds)
    min_V_droplet: Droplet = None
    for d in ld:
        if (min_V_droplet is None) or min_V_droplet.V > d.V:
            min_V_droplet = d
    return min_V_droplet



def shooting(
    V: float,
    A0: float,
    A1: float,
    B: float,
    a: float,
    stop_val:float = 1.0,
    tol: float = 1.0e-10,
    max_iter: int = 100,
    d_type: str = "pendant"
) -> float:
    def volume(A: float) -> float:
        droplets = find_droplets(A, B, a, np.pi * 5.0, 0.001, stop_val, d_type)

        if not droplets:
            raise RuntimeError(f"No droplet found for A={A}")

        # Select the droplet whose volume is closest to the target.
        candidates = droplets[:2]
        droplet = min(candidates, key=lambda d: abs(d.V - V))
        return droplet.V

    V0 = volume(A0)
    V1 = volume(A1)

    for _ in range(max_iter):
        F0 = V0 - V
        F1 = V1 - V

        # Stop when the volume residual is sufficiently small.
        if abs(F1) <= tol:
            return A1

        denominator = V1 - V0

        if abs(denominator) <= const.EPSILON:
            raise RuntimeError(
                "Secant method failed: consecutive volume residuals "
                "are too close."
            )

        A_next = A1 - F1 * (A1 - A0) / denominator

        if abs(A_next - A1) <= tol:
            return A_next

        A0, V0 = A1, V1
        A1 = max(A_next, 1e-5)
        V1 = volume(A1)

    raise RuntimeError(
        f"Shooting method did not converge after {max_iter} iterations."
    )