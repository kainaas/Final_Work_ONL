from sys import path

path.append("./../src")

from droplet import *
import numpy as np


gamma = 0.0728      #N/m
density = 1000.0    #kg/m³
theta_f = 2.0*np.pi/3.0
cappilary_length = np.sqrt(gamma / (density* const.g))
ds = 0.001
max_s = 5 * np.pi
B = 1.0

 
#A = \delta p_0 / \sqrt((\delta \rho g \gamma))


plot_type = "droplet" #may be "Vxd", "VxA", "droplet", "droplets", "Vstar", "max_press", "no_g", "various", "anim"
show_plot = True


if plot_type == "Vxd":
    lA = np.hstack([np.linspace(0.0, 1.0, 400), np.linspace(1.0, 100.0, 100)])
    lV = []
    ld = []
    for A in lA:
        droplets = find_droplets(A, B, cappilary_length, 5*np.pi, ds, theta_f, "sessile")
        for d in droplets:
            lV.append(d.V)
            ld.append(d.get_height())
    

    fig = plt.figure(figsize = (1000, 1000, "px"))
    ax = fig.add_subplot(111)
    ax.plot(lV, ld)
    ax.set_title(f"Volume x droplet height")
    ax.set_xlabel("Volume")
    ax.set_ylabel("d")


elif plot_type == "VxA": 
    fig = plt.figure(figsize = (1000, 1000, "px"))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")

    lA = np.hstack([np.linspace(0.0, 1.0, 400), np.linspace(1.0, 100.0, 100)])
    lV = []
    lA_plot = []
    for A in lA:
        droplets = find_droplets(A, B, cappilary_length, 5*np.pi, ds, theta_f, "sessile")
        for d in droplets:
            lV.append(d.V)
            lA_plot.append(A)
    

    fig = plt.figure(figsize = (1000, 1000, "px"))
    ax = fig.add_subplot(111)
    ax.plot(lV, lA_plot)
    ax.set_title(f"Volume x A")
    ax.set_xlabel("Volume")
    ax.set_ylabel("A")


elif plot_type == "droplet":
    V = 0.5
    A = shooting(V, 0.01, 4.0, B, cappilary_length, theta_f, d_type="sessile")
    print(A, V)
    # A = 0.00000275
    droplets = find_droplets(A, B, cappilary_length, 5*np.pi, ds, theta_f, "sessile")


    p0 = np.sqrt(density * const.g * gamma) * A
    fig = plt.figure(figsize=(1000,1000, 'px'))
    ax = fig.add_subplot(111)
    for i, droplet in enumerate(droplets):
        if i > 1: break
        droplet.plot(ax, f"Intersection {i+1}, V = {droplet.V:.2f}")
        
    ax.set_title(f"Droplet A = {A}")
    ax.legend()
    ax.set_aspect("equal")

if show_plot:
    plt.show()