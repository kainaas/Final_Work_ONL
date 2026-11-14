import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def cross(x1: np.array, x2: np.array):
    return x1[0]*x2[1] - x1[1]*x2[0]

def point(x1: float, x2: float) -> np.array:
    return np.array([x1, x2])

def angle_origin_vector(x: np.ndarray) -> float:
    theta = np.arctan2(x[1], x[0])
    if theta < 0:
        theta += 2*np.pi
    return theta



class Line:
    '''
        Represents a line in 2D. x1 and x2 represents the begin and end points of the line. 
        The normal vector is considering that objects are ran counter-clockwise. 
        The points list stores the points used in quadratures.
    '''
    def __init__(self, x1: np.array, x2: np.array, normal_counter_clock: bool = True):
        '''
            Initialize a line. The normal_counter_clock tells which of the two possible normals is the one used
        '''
        self.x1 = x1
        self.x2 = x2
        self.vec = x2 - x1
        self.normal_counter_clock = normal_counter_clock
        normal = np.array([self.vec[1],
                                -self.vec[0]
                                ])
        self.normal = normal/ np.linalg.norm(normal)
        if normal_counter_clock == False:
            self.invert_normal()

    def invert_normal(self): 
        '''
            When it's needed to invert the normal, i.e., points inside object. If the object is not simply conected
        '''
        for i in range(2):
            self.normal[i] = -self.normal[i]

    def translate(self, x_step: float, y_step: float):
        '''
            Move a line by relative coordinates (x_step, y_step)
        '''
        p = point(x_step, y_step)
        self.x1 = self.x1 + p
        self.x2 = self.x2 + p

    def rotate(self, matrix):
        '''
            Rotate a line by theta radians
        '''
        self.x1 = matrix @ self.x1
        self.x2 = matrix @ self.x2
        self.normal = matrix @ self.normal
        self.vec = self.x2 - self.x1

    def plot(self, ax: matplotlib.axes, normal: bool=False):
        '''
            Given the axes ax, plot the line in this ax. If normal is true, plot the normal arrow
        '''
        ax.plot([self.x1[0], self.x2[0]], [self.x1[1], self.x2[1]])
        if normal:
            midpoint = (self.x1 + self.x2)/2
            ax.quiver(midpoint[0], midpoint[1], self.normal[0], self.normal[1])

    #Parametrizes the line to the interval [-1, 1]
    def param_func(self, xi: float) -> np.array:
        return (1- xi)*self.x1 / 2 + (1 + xi)*self.x2 / 2







class Circle:
    '''
        Represents a full circle in 2D. 
        Has a lines list to store a discretized/segmented circle
    '''
    def __init__(self, center: np.array, radius: float, counter_clockwise: bool = True):
        self.c = center
        self.r = radius
        self.lines = []
        self.counter_clockwise = counter_clockwise

    def get_points_eq_space(self, n_segments: int):
        '''
            Gets the points used in a uniform segmentation of a circle in n_segments
        '''
        pace = 2*np.pi / (n_segments)
        return np.array([[self.c[0] + self.r*np.cos(i*pace), self.c[1] + self.r*np.sin(i*pace)] for i in range(n_segments+1)])

    def discretize(self, n_segments: int, counter_clockwise: bool = True):
        '''
            Discretize the circle in n_segments. These segments are appended in the lines list.
            Resets the existing lines list before discretizing
        '''
        self.lines = []
        points = self.get_points_eq_space(n_segments)
        for i in range(n_segments):
            l = Line(points[i,:], points[i+1,:], counter_clockwise)
            self.lines.append(l)

    def get_angular_size(self):
        '''
            The amount of radians that there is in a full circle
        '''
        return 2* np.pi

    def get_arc_length(self):
        '''
            Arc length of the circle
        '''
        return self.r * self.get_angular_size()

    def translate(self, x_step: float, y_step: float):
        '''
             Moves a circle by relative coordinates (x_step, y_step).
             All of its lines are moved too, if there are any
        '''
        p = point(x_step, y_step)
        self.c = self.c + p
        if len(self.lines) > 0:
            for l in self.lines:
                l.translate(x_step, y_step)

    def rotate(self, matrix):
        '''
            Rotate a circle by theta radians.
            All of its lines are moved too, if there are any
        '''
        self.c = matrix @ self.c
        if len(self.lines) > 0:
            for l in self.lines:
                l.rotate(matrix)

    def plot(self, ax: matplotlib.axes, n_line_segments: int = 100):
        '''
            Plots the circle using n_line_segments to approximate it
        '''
        points = self.get_points_eq_space(n_line_segments)
        ax.plot(points[:,0], points[:,1])

    def plot_discretized(self, ax: matplotlib.axes, normal: bool = False):
        '''
            Deprecated. Plot hte discretized circle. Substituded by the new Contour representation
        '''
        for i, l in enumerate(self.lines):
            l.plot(ax, normal)
    





class Arc(Circle):
    '''
        Specialization of a Circle. An arc of circle can be defined by an starting point, an end point, 
        a direction that the arc follows between the starting, a radius and an end point and a normal direction
    '''
    def __init__(self, x_start: np.array, x_end: np.array, center: np.array, counter_clockwise_normal: bool, counter_clockwise: bool = True):
        '''
            Starts the arc in the raw way.
        '''
        self.x1 = x_start
        self.x2 = x_end
        self.counter_clockwise_draw = counter_clockwise
        r = np.linalg.norm(center - x_start)
        super().__init__(center, r, counter_clockwise_normal)

    @classmethod
    def from_3_points(cls, x_start: np.array, x_middle: np.array, x_end: np.array, counter_clockwise_normal: bool = True):
        '''
            Starts an Arc using 3 points. The order of x_start, x_middle and x_end matters. 
        '''
        v1 = x_middle - x_start
        v2 = x_end - x_start
        cross_prod = cross(v1, v2)
        counter: bool = True
        if(np.abs(cross_prod) <= 1e-12): return None
        elif cross_prod < 0.0: counter = False
        
        x1 = x_start[0]
        x2 = x_middle[0]
        x3 = x_end[0]

        y1 = x_start[1]
        y2 = x_middle[1]
        y3 = x_end[1]
        
        M = np.array([[2*(x2 - x1), 2*(y2 - y1)],
                      [2*(x3 - x1), 2*(y3 - y1)]
                    ])
        b = np.array([x2*x2 + y2*y2 - x1*x1 - y1*y1, x3*x3 + y3*y3 - x1*x1 - y1*y1])
        
        center = np.linalg.solve(M, b)

        return cls(x_start, x_end, center, counter_clockwise_normal, counter)
    

    def get_angular_size(self):
        '''
            Overrides the get_angular_size() of Circle. Returns the angular size in radians of the Arc
        '''
        v1 = self.x1 - self.c
        v2 = self.x2 - self.c

        theta1 = angle_origin_vector(v1)
        theta2 = angle_origin_vector(v2)

        if self.counter_clockwise_draw:
            diff = (theta2 - theta1) % (2*np.pi)
        else:
            diff = (theta1 - theta2) % (2*np.pi)
        return diff

    def get_points_eq_space(self, n_segments: int):
        '''
            Gets the points used in a uniform segmentation of the arc.
        '''
        diff = self.get_angular_size()

        if diff < 1e-12: 
            super().plot(ax, n_segments)
            return

        v1 = self.x1 - self.c
        theta1 = angle_origin_vector(v1)

        theta = np.linspace(0, diff, n_segments + 1)

        if self.counter_clockwise_draw:
            theta += theta1
        else:
            theta = theta1 - theta

        return np.column_stack([
            self.c[0] + self.r*np.cos(theta),
            self.c[1] + self.r*np.sin(theta)
        ])


    def translate(self, x_step: float, y_step: float):
        p = point(x_step, y_step)
        self.x1 = self.x1 + p
        self.x2 = self.x2 + p
        self.c = self.c + p
        if len(self.lines) > 0:
            for l in self.lines:
                l.translate(x_step, y_step)

    def rotate(self, matrix):
        self.c = matrix @ self.c
        self.x1 = matrix @ self.x1
        self.x2 = matrix @ self.x2
        if len(self.lines) > 0:
            for l in self.lines:
                l.rotate(matrix)

        
        

def param(Lmatrix, xi:float):
    ''' 
        Returns the linear parametrization of each line in the matrix of a Contour, with values in the interval [-1,1].
        The returned matrix has dimensions (2, num_lines).
        The structure used in this matrix is described in the contour class
    '''
    tmp = np.zeros((2, len(Lmatrix)))
        
    tmp = (1-xi)/2* Lmatrix[0:2, :] + (1+xi)/2 * Lmatrix[3:5, :]

    return tmp





class Contour:
    '''
        A set of lines, arcs and circles. It's the representation of an object. Has a list of these components
        and a list of lines. The list of lines holds all discretization lines and natural object lines.
        Holds a referential position and angle.

        The lines of discretization are also represented in a matrix. this matrix has shape (6, num_lines).
        The first two rows are the beginning points of the lines. The 4th and 5th rows are the ending points
        of each line. The 3rd and 6th rows are the homogenous coordinate for the beggining and ending points respectively.
        This allows for more optimized rigid transformations.

        The normals are also stored this way, in a matrix of shape (3, num_lines).
    '''
    def __init__(self, density: float = 1.0):
        self.density = density
        self.components = []
        self.lines = []
        self.pos = np.zeros((2,))
        self.theta = 0.0
        self.Lmatrix: np.array = None
        self.Lnormals: np.array = None

    def add_component(self, component):
        '''
            Adds a component to the Contour list of components
        '''
        self.components.append(component)


    def _assemble_lines_matrix(self):
        ''' 
            First 2 lines are the coordinates of first point of each line. 
            Fourth and fifth line are the coordinates of the second point of each line.
            This structure makes us able to apply a rigid transformation as a unique matrix multiplication
        '''
        size = len(self.lines)
        self.Lmatrix = np.ones((6, size)) 
        self.Lnormals = np.zeros((2,size))
        for i, l in enumerate(self.lines):
            self.Lmatrix[0:2, i] = l.x1
            self.Lmatrix[3:5, i] = l.x2
            self.Lnormals[:, i] = l.normal
        
    def discretize_n_lines(self, n_segments: int):
        '''
            Discretize each possible component with n_segments each
        '''
        self.lines = []
        for _, comp in enumerate(self.components):
            if isinstance(comp, Circle):
                comp.discretize(n_segments, comp.counter_clockwise)
                for _, l in enumerate(comp.lines):
                    self.lines.append(l)

            else: self.lines.append(comp)
        self._assemble_lines_matrix()

    def discretize_size_lines(self, max_size: float):
        '''
            Discretize the lines using a max_size
        '''
        self.lines = []
        for _, comp in enumerate(self.components):
            if isinstance(comp, Circle):
                arc_length = comp.get_arc_length()
                n_segments = arc_length / max_size
                if n_segments.is_integer():
                    n_segments = int(n_segments)
                else:    
                    n_segments = int(n_segments+1)
                comp.discretize(n_segments, comp.counter_clockwise)
                for _, l in enumerate(comp.lines):
                    self.lines.append(l)

            else: self.lines.append(comp)
        self._assemble_lines_matrix()


    def param(self, xi: float) -> np.array:
        ''' 
            Returns the parametrization of each line in the contour, with values in the interval [-1,1].
            The returned matrix has dimensions (2, num_lines).
        '''
        return param(self.Lmatrix, xi)

    def plot(self, ax: matplotlib.axes, n_line_segments: int = 100, normal:bool = False):
        '''
            Plots each component. Kind of deprecated. Can plot the position and rotation wrong if they are not used
            thinking in plot the non-discretized object
        '''
        for _, comp in enumerate(self.components):
            if isinstance(comp, Circle):
                comp.plot(ax, n_line_segments)
            else: comp.plot(ax, normal)

    def plot_discretized_original(self, ax: matplotlib.axes, normal: bool = False):
        '''
            Plots each line of the discretized object. Deprecated.
        '''
        for _, l in enumerate(self.lines):
            l.plot(ax, normal)

    def plot_discretized(self, ax: matplotlib.axes):
        '''
            Plots the discretized object
        '''
        x = []
        y = []
        for i in range(len(self.lines)):
            x.append(self.Lmatrix[0, i])
            x.append(self.Lmatrix[3, i])
            y.append(self.Lmatrix[1, i])
            y.append(self.Lmatrix[4, i])
        return ax.plot(x, y)


    def translate(self, x_step: float, y_step: float, translate_abstraction: bool = False):
        '''
            Moves the matrix representation of the object by (x_step, y_step). 
            If translate_abstraction is true, also move all components objects
        '''
        p = point(x_step, y_step)
        self.pos = self.pos + p

        if translate_abstraction:
            for comp in self.components:
                comp.translate(x_step, y_step)
        
        if not self.Lmatrix is None:
            M = np.array([
                [1.0, 0.0, x_step],
                [0.0, 1.0, y_step],
                [0.0, 0.0, 1.0]
            ])
            self.Lmatrix = np.vstack([M @ self.Lmatrix[0:3, :], M @ self.Lmatrix[3:, :]])


    def rotate(self, theta: float, rotate_abstraction: bool = False):
        '''
            Rotates the matrix representation of the object by theta radians. 
            If translate_abstraction is true, also rotate all components objects
        '''
        s = np.sin(theta)
        c = np.cos(theta)
        self.theta += theta
        if rotate_abstraction:
            mat = np.array([[c, -s], [s, c]])
            for comp in self.components:
                comp.rotate(mat)

        if not self.Lmatrix is None:
            M = np.array([
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0]
            ])
            self.Lmatrix = np.vstack([M @ self.Lmatrix[0:3, :], M @ self.Lmatrix[3:, :]])
            self.Lnormals = M[0:2, 0:2] @ self.Lnormals

    def apply_RT(self, matrix, rotation, x_step_total, y_step_total, theta):
        ''' 
            Apply a transformation composed f rotations and translations. The transformation is given in matrix.
            Must input the x_step_total, y_step_total and theta of the transformation, so the position and angle
            variables can be refreshed
        '''
        p = point(x_step_total, y_step_total)
        self.pos = self.pos + p
        self.theta += theta
        self.Lmatrix = np.vstack([matrix @ self.Lmatrix[0:3, :], matrix @ self.Lmatrix[3:, :]])
        self.Lnormals = rotation[0:2, 0:2] @ self.Lnormals


    def reset_position(self):
        self.pos = np.zeros((2,))

    def reset_angle(self):
        self.theta = 0.0

    def find_biggest_dist(self):
        ''' 
            Naive implementation to find the biggest distance betwwen two points of the contour. O(N²)
        '''
        maxm= 0.0
        n_points = np.shape(self.Lmatrix)[1]
        for i in range(n_points):
            for j in range(i+1, n_points):
                maxm =  max(maxm, np.linalg.norm(self.Lmatrix[0:2, i] - self.Lmatrix[0:2, j]))
        return maxm

    @classmethod
    def read_file(cls, file, density:float = 1.0):
        '''
            File example in objects/example.txt
        '''
        ctr = cls(density)
        with open(file, 'r') as f:
            for line in f:
                counter_clockwise = 1
                l = line.split()
                if len(l) == 0:
                    pass
                elif l[0] == "#":
                    pass
                elif l[0] == "L":
                    x1, y1, x2, y2 = float(l[1]), float(l[2]), float(l[3]), float(l[4])
                    start = point(x1, y1)
                    end = point(x2, y2)
                    if len(l) > 5:
                        counter_clockwise = l[5] == "1"
                    ctr.add_component(Line(start, end, counter_clockwise))

                elif l[0] == "C":
                    c1, c2 = float(l[1]), float(l[2])
                    r = float(l[3])
                    if len(l) > 4:
                        counter_clockwise = l[4] == "1"
                    center = point(c1,c2)
                    ctr.add_component(Circle(center, r, counter_clockwise))

                elif l[0] == "Ac":
                    x1, y1, x2, y2, c1, c2 = float(l[1]), float(l[2]), float(l[3]), float(l[4]), float(l[5]), float(l[6])
                    if len(l) > 7:
                        counter_clockwise = l[7] == "1"
                    counter_clockwise_draw = True
                    if len(l) > 8:
                        counter_clockwise_draw = l[8] == "1"
                    start = point(x1, y1)
                    end = point(x2, y2)
                    center =point(c1, c2)
                    ctr.add_component(Arc(start, end, center, counter_clockwise, counter_clockwise_draw))

                elif l[0] == "A3":
                    x1, y1, x2, y2, x3, y3 = float(l[1]), float(l[2]), float(l[3]), float(l[4]), float(l[5]), float(l[6])
                    start = point(x1, y1)
                    mid = point(x2, y2)
                    end = point(x3, y3)
                    ctr.add_component(Arc.from_3_points(start, mid, end))
        return ctr

    @classmethod
    def read_point_list(cls, file, density: float = 1.0, counter_clockwise: bool = True):
        '''
            Reads a point list and assemble a contour with this line list. counter_clockwise dictates the
            direction the points are wandered. The points must be sequential. Does not suport objects with holes.
        '''
        ctr = cls(density)
        curr_line = []
        next_line = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)-1):
                l1 = lines[i].replace("\ufeff", "").split()
                l2 = lines[i+1].replace("\ufeff", "").split()
                p1 = point(float(l1[0]), float(l1[1]))
                p2 = point(float(l2[0]), float(l2[1]))
                ctr.add_component(Line(p1, p2, counter_clockwise))
            
            return ctr
                






if __name__ == "__main__":
    obj = Contour.read_file("objects/square.txt")
    
    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = fig.add_subplot(111)
    p1 = point(0.0,0.0)
    print(type(p1[1]))
    obj.discretize_n_lines(10)
    obj.translate(-1.0, -1.0)
    obj.rotate(np.pi/4)
    print(obj.Lnormals)
    
    d = True
    if d:
        obj.plot_discretized(ax)
    else:
        obj.plot(ax)

    ax.set_aspect('equal', adjustable='box')
    plt.show()
