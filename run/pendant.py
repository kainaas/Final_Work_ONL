from sys import path

path.append("./../src/2D")

from droplet import * 


'''
Max volume droplet:
    A = 1.19721119513
    V* = 35.39486160126

Max pressure droplet:
    A* = 2.04614459832
    V = 2.17221828749
'''

gamma = 0.0728      #N/m
density = 1000.0    #kg/m³
a = 0.001           #m

ds = 0.001
max_s = 5 * np.pi
B = density * const.g * a * a / gamma


plot_type = "anim" #may be "Vxd", "VxA", "droplet", "droplets", "Vstar", "max_press", "no_g", "various", "anim"
show_plot = True


if plot_type == "Vxd":
    lA = np.linspace(0.0, 3.0, 500)
    lV = []
    ld = []
    for A in lA:
        droplets = find_droplets(A, B, a, 5*np.pi, 0.001)
        for d in droplets:
            lV.append(d.V)
            ld.append(d.get_height())
    

    fig = plt.figure(figsize = (1000, 1000, "px"))
    ax = fig.add_subplot(111)
    ax.scatter(lV, ld)
    ax.set_title(f"Volume x droplet height")
    ax.set_xlabel("Volume")
    ax.set_ylabel("d")
    
elif plot_type == "VxA": 
    fig = plt.figure(figsize = (1000, 1000, "px"))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")

    lA = np.linspace(0.0, 3.0, 500)
    lV = []
    lA_plot = []
    for A in lA:
        droplets = find_droplets(A, B, a, 5*np.pi, 0.001)
        for d in droplets:
            lV.append(d.V)
            lA_plot.append(A)
    

    fig = plt.figure(figsize = (1000, 1000, "px"))
    ax = fig.add_subplot(111)
    ax.scatter(lV, lA_plot)
    ax.set_title(f"Volume x A")
    ax.set_xlabel("Volume")
    ax.set_ylabel("A")



elif plot_type == "droplet":
    V = 7.824999999999999
    A = shooting(V, 1.5, 2.0, B, a)
    print(A, V)
    droplets = find_droplets(A, B, a, np.pi * 5, 0.001)


    p0 = (gamma/a) * A
    fig = plt.figure(figsize=(1000,1000, 'px'))
    ax = fig.add_subplot(111)
    for i, droplet in enumerate(droplets):
        if i > 1: break
        droplet.plot(ax, f"Intersection {i+1}, V = {droplet.V:.2f}")
        
    ax.set_title(f"Droplets A = {A}")
    ax.legend()
    ax.set_aspect("equal")




elif plot_type == "Vstar":
    
    def V_droplet(A):
        return max_V_droplet(A, B, a, np.pi * 5.0, 0.001)
    
    tol = 1.0e-11
    
    A_min = 1.19
    A_max = 1.21
    while A_max - A_min > tol:
        print(A_min, A_max)
        A_l = A_min + (A_max - A_min) / 3
        A_r = A_max - (A_max - A_min) / 3

        V_l = V_droplet(A_l)
        V_r = V_droplet(A_r)

        if V_l.V < V_r.V:
            A_min = A_l
        else:
            A_max = A_r

    A_max_val = (A_min + A_max) / 2.0
    max_droplet= V_droplet(A_max_val)

    p0 = (gamma/a) * A_max_val
    fig = plt.figure(figsize=(1000,1000, 'px'))
    ax = fig.add_subplot(111)
    max_droplet.plot(ax)
    ax.set_title("Max volume droplet")
    ax.set_aspect("equal")
    ax.text(
        0.5, 0.5,
        f"p_0 ={p0}\n A={A_max_val}\nV*={max_droplet.V}",
        transform=fig.transFigure,
        fontsize=14, ha='center')


elif plot_type == "max_press":

    def V_droplet(A):
        return min_V_droplet(A, B, a, np.pi * 5.0, 0.001)
    tol = 1.0e-11
    
    A_min = 2.0
    A_max = 2.2
    while A_max - A_min > tol:
        print(A_min, A_max)
        A_m = (A_max + A_min) / 2

        V_m = V_droplet(A_m)

        if V_m.V > 5.0:
            A_max = A_m
        else:
            A_min = A_m

    A_max_val = A_min
    max_droplet = V_droplet(A_max_val)

    p0 = (gamma/a) * A_max_val
    fig = plt.figure(figsize=(1000,1000, 'px'))
    ax = fig.add_subplot(111)
    max_droplet.plot(ax)
    ax.set_title("Max pressure droplet")
    ax.set_aspect("equal")
    ax.text(
        0.5, 0.5,
        f"p_0 ={p0}\n A={A_max_val}\nV*={max_droplet.V}",
        transform=fig.transFigure,
        fontsize=14, ha='center')


elif plot_type == "no_g":
    g = 0.0
    B = 0.0
    A = 1.0
    droplets = find_droplets(A, B, a, np.pi * 5, 0.001)


    p0 = (gamma/a) * A
    fig = plt.figure(figsize=(1000,1000, 'px'))
    ax = fig.add_subplot(111)
    for i, droplet in enumerate(droplets):
        if i > 1: break
        droplet.plot(ax, f"Intersection {i+1}, V = {droplet.V:.2f}")
        if i == 1:
            height = droplet.get_height()
            radius = 2.0 / A 
            height -= radius
            c = Circle(point(0.0, -height), radius)
            c.plot(ax)

        
    ax.set_title(f"Droplets A = {A}")
    ax.legend()
    ax.set_aspect("equal")

elif plot_type == "various" or plot_type == "anim":
    max_V = 31.0
    min_V = 0.1
    num_d = 19
    lV = np.linspace(min_V, max_V, num_d)
            
    fig = plt.figure(figsize=(1000, 1000, 'px'))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    droplets = []
    for V in lV:
        A: float
        print("V=", V)
        if V <= 2.17221828749:
            A = shooting(V, 0.0, 2.0, B, a)
        elif V > 20.0:
            A = shooting(V, 1.2, 1.4, B, a)
        elif V > 10.0:
            A = shooting(V, 1.3, 1.6, B, a)
        elif V > 5.0:
            A = shooting(V, 1.5, 1.9, B, a)
        elif V > 2.17221828749:
            A = shooting(V, 1.2, 1.9, B, a)
        
        ld = find_droplets(A, B, a, np.pi * 5.0, 0.001)
        if len(ld) >= 2:
            ld = ld[:2]
        d = sorted(ld, key=lambda d: abs(d.V - V))[0]
        droplets.append(d)

    droplets.append(find_droplets(1.19721119513, B, a, np.pi*5.0, 0.001)[1])

    if plot_type == "various":
        for d in droplets:
            d.plot(ax, label=f"V={d.V:.3f}")
        ax.legend()



    if plot_type == "anim":
        save = "imgs/test.gif"

        duration = 2.0 #in seconds
        interval = int(duration/len(droplets) * 1000) #in ms

        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-5.5, 0.1)

        line, = ax.plot([], [], lw=2)
        line.set_color("black")


        def init():
            line.set_data([], [])
            return (line,)


        def update(frame):
            x, y = droplets[frame].get_plot_vecs()
            line.set_data(x, y)

            return (line,)



        anim = FuncAnimation(
            fig,
            update,
            frames=len(droplets),
            init_func=init,
            interval=interval,
            blit=True
        )
        if not (save is None):
            anim.save(save)

if show_plot:
    plt.show()