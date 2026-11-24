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




def _rule(QuadRule: str):
    '''
        Returns (points (P, 2), weights (P,)) of QuadRule on the master triangle.
    '''
    if QuadRule not in Quadrature.points:
        raise ValueError(f"Unknown quadrature rule: {QuadRule}")
    return np.asarray(Quadrature.points[QuadRule]), Quadrature.weights[QuadRule]


def _triangle_geometry(x: np.array, mesh: Mesh, center: np.array, QuadRule: str):
    '''
        Geometry of every triangle of the deformed mesh at every quadrature point.

        Returns
        -------
        T : ndarray (F, P, 3)
            Position T(xi) = a + xi_0 (b - a) + xi_1 (c - a) of each quadrature point.
        normals : ndarray (F, 3)
            Unit outward normals (constant on each flat triangle).
        G : ndarray (F,)
            Jacobian of the map (twice the triangle area).
        weights : ndarray (P,)
            Quadrature weights (they sum to 1).
    '''
    xi, weights = _rule(QuadRule)
    m = deformed_mesh(x, mesh, center)
    normals, areas = m.calculate_face_normals()

    triangles = np.asarray(m.points)[np.asarray(m.connectivity, dtype=int)]
    a, b, c = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
    T = (a[:, None, :]
         + xi[None, :, 0, None] * (b - a)[:, None, :]
         + xi[None, :, 1, None] * (c - a)[:, None, :])
    return T, normals, 2.0 * areas, weights


def energy_3D(x: np.array, mesh: Mesh, Bond: float, center: np.array,
              QuadRule: str = "strang3") -> float:
    '''
        E = area + (Bond/2) * integral of z^2 n_z dA, on the deformed mesh.
    '''
    T, normals, G, w = _triangle_geometry(x, mesh, center, QuadRule)
    integrand = 1.0 + (Bond / 2.0) * normals[:, None, 2] * T[..., 2]**2
    # master triangle has area 1/2 and the weights sum to 1
    return float(np.sum(G * (integrand @ w)) / 2.0)


def volume_3D(x: np.array, mesh: Mesh, center: np.array,
              QuadRule: str = "strang3") -> float:
    '''
        V = (1/3) * integral of T . n dA (divergence theorem), on the deformed mesh.
        The surface is open at the nozzle; this is only correct because the
        nozzle lies on the plane z = 0 and T . n vanishes there.
    '''
    T, normals, G, w = _triangle_geometry(x, mesh, center, QuadRule)
    integrand = np.einsum("fpk,fk->fp", T, normals)
    return float(np.sum(G * (integrand @ w)) / 6.0)
