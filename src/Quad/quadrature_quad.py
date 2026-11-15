import numpy as np
from mesh_quad import QuadMesh, deformed_quad_mesh


def quad_gauss(n=3):
    """Gauss-Legendre tensorial n x n no quadrado [0,1]^2 -> (xi, eta, w), cada um com n*n entradas."""
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    XI, ETA = np.meshgrid(x, x, indexing="ij")
    return XI.ravel(), ETA.ravel(), np.outer(w, w).ravel()


def quad_geometry(points, faces, xi, eta):
    """
    Mapa bilinear de cada quad. Retorna, em cada ponto de Gauss:
      T    (F, P, 3): posicao
      nvec (F, P, 3): dT/dxi x dT/deta  (= n_unitaria * dA/(dxi deta))
    Como a face de um quad geralmente NAO e plana, a normal varia dentro dela
    e por isso e calculada em cada ponto de Gauss.
    """
    q = points[faces]                       # (F, 4, 3)
    q0, q1, q2, q3 = (q[:, k, :][:, None, :] for k in range(4))
    xi = xi[None, :, None]
    eta = eta[None, :, None]

    T = (1-xi)*(1-eta)*q0 + xi*(1-eta)*q1 + xi*eta*q2 + (1-xi)*eta*q3
    dxi = (1-eta)*(q1-q0) + eta*(q2-q3)
    deta = (1-xi)*(q3-q0) + xi*(q2-q1)
    return T, np.cross(dxi, deta)


def energy_3D_quad(x, mesh, Bond, center, n_gauss=3):
    """E = area + (Bond/2) * integral de z^2 n_z dA"""
    m = deformed_quad_mesh(x, mesh, center)
    pts = np.asarray(m.points, float)
    faces = np.asarray(m.connectivity, int)
    xi, eta, w = quad_gauss(n_gauss)

    T, nvec = quad_geometry(pts, faces, xi, eta)
    dA = np.linalg.norm(nvec, axis=2)
    integrand = dA + 0.5 * Bond * T[..., 2]**2 * nvec[..., 2]   # n_z dA = nvec_z
    return float(np.sum(integrand * w[None, :]))


def volume_3D_quad(x, mesh, center, n_gauss=3):
    """V = (1/3) * integral de T . n dA  (teorema da divergencia)"""
    m = deformed_quad_mesh(x, mesh, center)
    pts = np.asarray(m.points, float)
    faces = np.asarray(m.connectivity, int)
    xi, eta, w = quad_gauss(n_gauss)

    T, nvec = quad_geometry(pts, faces, xi, eta)
    integrand = np.einsum("fpk,fpk->fp", T, nvec)
    return float(np.sum(integrand * w[None, :]) / 3.0)
