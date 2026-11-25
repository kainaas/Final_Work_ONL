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


# ---------------------------------------------------------------------------
# exact gradients with respect to the radial displacements x
# ---------------------------------------------------------------------------

def _free_ids_and_directions(x: np.array, mesh: Mesh, center: np.array):
    '''
        Free vertex ids and the direction d_i = P_i - center along which each
        one moves (dP_i/dx_i = d_i).
    '''
    x = np.asarray(x, dtype=float).ravel()
    if mesh.free_ids is None:
        ids = np.arange(len(x))
    else:
        ids = np.asarray(mesh.free_ids, dtype=int)
    points0 = np.asarray(mesh.points, dtype=float)
    return ids, points0[ids] - np.asarray(center, dtype=float).reshape(1, 3)


def _chain_rule(grad_a, grad_b, grad_c, x, mesh, center):
    '''
        Adds the per-triangle gradients with respect to the vertices a, b, c
        into a gradient per vertex and projects it on the free directions.
    '''
    faces = np.asarray(mesh.connectivity, dtype=int)
    grad_points = np.zeros((mesh.num_points, 3))
    np.add.at(grad_points, faces[:, 0], grad_a)
    np.add.at(grad_points, faces[:, 1], grad_b)
    np.add.at(grad_points, faces[:, 2], grad_c)
    ids, d = _free_ids_and_directions(x, mesh, center)
    return np.einsum("ik,ik->i", grad_points[ids], d)


def _deformed_triangles(x, mesh, center):
    m = deformed_mesh(x, mesh, center)
    tri = np.asarray(m.points)[np.asarray(m.connectivity, dtype=int)]
    return tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]


def grad_energy_3D(x: np.array, mesh: Mesh, Bond: float, center: np.array,
                   QuadRule: str = "strang3") -> np.ndarray:
    '''
        Exact gradient of energy_3D with respect to x.

        With u = b - a, v = c - a and N = u x v (= 2 A n), each triangle
        contributes |N|/2 + (Bond/4) N_z S, where S = sum_p w_p T_z(xi_p)^2.
        The quadrature is applied to the same integrand as in energy_3D, so
        this is the derivative of the discrete energy (for any rule).
    '''
    xi, w = _rule(QuadRule)
    a, b, c = _deformed_triangles(x, mesh, center)
    u, v = b - a, c - a
    N = np.cross(u, v)
    normN = np.linalg.norm(N, axis=1)
    n = N / normN[:, None]

    # |N|/2: d|N| = n . (du x v + u x dv)
    gb_area = 0.5 * np.cross(v, n)
    gc_area = 0.5 * np.cross(n, u)
    ga_area = -(gb_area + gc_area)

    # N_z: same formula with n replaced by e_z
    ez = np.array([0.0, 0.0, 1.0])
    gb_Nz = np.cross(v, ez)
    gc_Nz = np.cross(ez, u)
    ga_Nz = -(gb_Nz + gc_Nz)

    # S = sum_p w_p T_z^2, T_z = (1 - xi0 - xi1) a_z + xi0 b_z + xi1 c_z
    l0 = 1.0 - xi[:, 0] - xi[:, 1]
    Tz = (np.outer(a[:, 2], l0) + np.outer(b[:, 2], xi[:, 0]) + np.outer(c[:, 2], xi[:, 1]))
    S = (Tz**2) @ w
    dS_da = (2.0 * Tz * l0) @ w
    dS_db = (2.0 * Tz * xi[:, 0]) @ w
    dS_dc = (2.0 * Tz * xi[:, 1]) @ w

    k = Bond / 4.0
    Nz = N[:, 2]

    def grav(gNz, dS):
        g = k * S[:, None] * gNz
        g[:, 2] += k * Nz * dS
        return g

    return _chain_rule(ga_area + grav(ga_Nz, dS_da),
                       gb_area + grav(gb_Nz, dS_db),
                       gc_area + grav(gc_Nz, dS_dc), x, mesh, center)


def grad_volume_3D(x: np.array, mesh: Mesh, center: np.array,
                   QuadRule: str = "strang3") -> np.ndarray:
    '''
        Exact gradient of volume_3D with respect to x.

        On a flat triangle T . N is constant (u . N = v . N = 0), so for any
        rule whose weights sum to 1 the volume is sum a . (b x c) / 6 and its
        derivatives are (b x c)/6, (c x a)/6 and (a x b)/6.
    '''
    _rule(QuadRule)  # only validates the name
    a, b, c = _deformed_triangles(x, mesh, center)
    return _chain_rule(np.cross(b, c) / 6.0, np.cross(c, a) / 6.0,
                       np.cross(a, b) / 6.0, x, mesh, center)
