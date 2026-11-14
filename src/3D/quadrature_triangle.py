import numpy as np

from mesh import *


def point_2D(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=float)
#https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
class Quadrature:
    '''
        This class holds the implementation of quadrature.
        Supports quadratures on triangle with vertices (0,0), (1,0), (0,1). Quadratrure rules supported:
        strang3:  4 points,  exact for polynomials of degree <= 3
        strang7:  7 points,  exact for polynomials of degree <= 5
        strang10: 13 points, exact for polynomials of degree <= 7
        (the number in the name comes from the tabulated rule, it is not the
        number of points). Weights are normalised to sum to 1, so the integral
        over the master triangle (area 1/2) is  sum(w f(x)) / 2.
    '''
    points = {"strang3":  [point_2D(0.33333333333333333333, 0.33333333333333333333),
                           point_2D(0.60000000000000000000, 0.20000000000000000000),
                           point_2D(0.20000000000000000000, 0.60000000000000000000),
                           point_2D(0.20000000000000000000, 0.20000000000000000000)],

              "strang7":  [point_2D(0.33333333333333333, 0.33333333333333333),
                           point_2D(0.79742698535308720, 0.10128650732345633),
                           point_2D(0.10128650732345633, 0.79742698535308720),
                           point_2D(0.10128650732345633, 0.10128650732345633),
                           point_2D(0.05971587178976981, 0.47014206410511505),
                           point_2D(0.47014206410511505, 0.05971587178976981),
                           point_2D(0.47014206410511505, 0.47014206410511505)],
              
              "strang10": [point_2D(0.333333333333333, 0.333333333333333),
                           point_2D(0.479308067841923, 0.260345966079038),
                           point_2D(0.260345966079038, 0.479308067841923),
                           point_2D(0.260345966079038, 0.260345966079038),
                           point_2D(0.869739794195568, 0.065130102902216),
                           point_2D(0.065130102902216, 0.869739794195568),
                           point_2D(0.065130102902216, 0.065130102902216),
                           point_2D(0.638444188569809, 0.312865496004875),
                           point_2D(0.638444188569809, 0.048690315425316),
                           point_2D(0.312865496004875, 0.638444188569809),
                           point_2D(0.312865496004875, 0.048690315425316),
                           point_2D(0.048690315425316, 0.638444188569809),
                           point_2D(0.048690315425316,0.312865496004875)],
             }

    weights = { "strang3":  np.array([-0.56250000000000000000,
                                      0.52083333333333333333,
                                      0.52083333333333333333,
                                      0.52083333333333333333]),
                             
                "strang7":  np.array([0.22500000000000000,
                                      0.12593918054482717,
                                      0.12593918054482717,
                                      0.12593918054482717,
                                      0.13239415278850616,
                                      0.13239415278850616,
                                      0.13239415278850616]),

                "strang10": np.array([-0.149570044467670,
                                      0.175615257433204,
                                      0.175615257433204,
                                      0.175615257433204,
                                      0.053347235608839,
                                      0.053347235608839,
                                      0.053347235608839,
                                      0.077113760890257,
                                      0.077113760890257,
                                      0.077113760890257,
                                      0.077113760890257,
                                      0.077113760890257,
                                      0.077113760890257])
              }


    @classmethod
    def integrate(cls, f, QuadRule: str = "strang10") -> float:
        '''
            Integrate f using QuadRule. It can be "strang3", "strang7" or "strang10"
        '''
        if QuadRule not in cls.points:
            raise ValueError(f"Unknown quadrature rule: {QuadRule}")

        points = cls.points[QuadRule]
        weights = cls.weights[QuadRule]

        if len(points) != len(weights):
            raise ValueError("Number of points and weights must match.")

        values = np.array([f(p) for p in points])

        return float(np.dot(weights, values) / 2.0)




def energy_3D(x: np.array, mesh: Mesh, Bond: float, center: np.array) -> float:
    m = deformed_mesh(x, mesh, center)
    
    normals, areas = m.calculate_face_normals()
    G = 2.0 * areas

    points = np.asarray(mesh.points)
    connectivity = np.asarray(mesh.connectivity, dtype=int)

    # Construct triangular faces
    triangles = points[connectivity]
    N = len(m.connectivity)
    # triangles[i,j,:] #[triangle, point, :(whole point)]

    sumE = 0.0

    for i in range(N):
        triang = triangles[i]
        a = triang[0]
        b = triang[1]
        c = triang[2]
        Gi = G[i]
        normal = normals[i]
        T = lambda x: x[0] * (b-a) + x[1] * (c-a) + a
        f = lambda x:  Gi * (1 + (Bond / 2.0) * normal[2] * T(x)[2]**2)
        sumE += Quadrature.integrate(f, "strang3")

    return sumE

def volume_3D(x: np.array, mesh: Mesh, center:np.array) -> float:
    m = deformed_mesh(x, mesh, center)
    
    normals, areas = m.calculate_face_normals()
    G = 2.0 * areas

    points = np.asarray(mesh.points)
    connectivity = np.asarray(mesh.connectivity, dtype=int)

    # Construct triangular faces
    triangles = points[connectivity]
    N = len(m.connectivity)
    # triangles[i,j,:] #[triangle, point, :(whole point)]

    sumV = 0.0
    for i in range(N):
        triang = triangles[i]
        a = triang[0]
        b = triang[1]
        c = triang[2]
        Gi = G[i]
        normal = normals[i]
        T = lambda x: x[0] * (b-a) + x[1] * (c-a) + a
        f = lambda x: Gi * np.dot(T(x), normal)
        sumV += Quadrature.integrate(f, "strang3")

    return sumV / 3.0