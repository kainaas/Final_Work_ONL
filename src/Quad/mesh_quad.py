import numpy as np
from mesh import Mesh, point, plot_mesh   # reaproveita resample_contour e plot_mesh
class QuadMesh(Mesh):
    """
    Malha de quadrilateros. Cada face e (q0, q1, q2, q3) com a convencao

        q3 ---- q2
        |        |        xi  : q0 -> q1
        q0 ---- q1        eta : q0 -> q3

    e a normal (dT/dxi x dT/deta) aponta PARA FORA da gota.
    Na calota esferica o ultimo anel liga ao polo com um quad degenerado
    (q1 = q2 = polo), que se comporta como um triangulo.
    """

    def add_quad(self, i, j, k, l):
        self.connectivity.append((i, j, k, l))

    @classmethod
    def cut_sphere(cls, r, num_sectors, num_stacks, min_theta, z_center):
        if r <= 0:
            raise ValueError("Radius must be positive.")
        if num_sectors < 3 or num_stacks < 1:
            raise ValueError("Invalid mesh resolution.")
        if not 0 < min_theta < np.pi:
            raise ValueError("min_theta must belong to (0, pi).")

        mesh = cls()
        theta_values = np.linspace(min_theta, np.pi, num_stacks + 1)
        phi_values = np.linspace(0, 2 * np.pi, num_sectors, endpoint=False)

        for theta in theta_values[:-1]:
            for phi in phi_values:
                mesh.add_point(point(
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta) + z_center))

        pole_id = mesh.add_point(point(0, 0, -r + z_center))
        mesh.cut_boundary = list(range(num_sectors))

        # aneis regulares: 1 quad no lugar de 2 triangulos
        for i in range(num_stacks - 1):
            o0, o1 = i * num_sectors, (i + 1) * num_sectors
            for j in range(num_sectors):
                j1 = (j + 1) % num_sectors
                mesh.add_quad(o0 + j, o1 + j, o1 + j1, o0 + j1)

        # ultimo anel -> polo (quad degenerado)
        o = (num_stacks - 1) * num_sectors
        for j in range(num_sectors):
            j1 = (j + 1) % num_sectors
            mesh.add_quad(o + j, pole_id, pole_id, o + j1)

        return mesh

    def attach_nozzle(self, nozzle_contour, z_nozzle, nozzle_length=0.0):
        if not self.cut_boundary:
            raise ValueError("No spherical cut boundary found.")
        if nozzle_length < 0:
            raise ValueError("nozzle_length must be nonnegative.")

        sphere_ids = np.asarray(self.cut_boundary)
        sphere_ring = np.asarray(self.points)[sphere_ids]
        if z_nozzle <= sphere_ring[0, 2]:
            raise ValueError("Nozzle must be above the spherical cut.")

        n = len(sphere_ids)
        nozzle_ring = self.resample_contour(nozzle_contour, n)

        costs = [np.sum((sphere_ring[:, :2] - np.roll(nozzle_ring, s, axis=0))**2)
                 for s in range(n)]
        nozzle_ring = np.roll(nozzle_ring, int(np.argmin(costs)), axis=0)

        nozzle_ids = [self.add_point(point(x, y, z_nozzle)) for x, y in nozzle_ring]

        for j in range(n):
            j1 = (j + 1) % n
            self.add_quad(sphere_ids[j], sphere_ids[j1], nozzle_ids[j1], nozzle_ids[j])

        if nozzle_length > 0:
            top_ids = [self.add_point(point(x, y, z_nozzle + nozzle_length))
                       for x, y in nozzle_ring]
            for j in range(n):
                j1 = (j + 1) % n
                self.add_quad(nozzle_ids[j], nozzle_ids[j1], top_ids[j1], top_ids[j])
            self.nozzle_boundary = top_ids
        else:
            self.nozzle_boundary = nozzle_ids

        self.has_nozzle = True
        return self

    def calculate_face_normals(self):
        """Normais unitarias e areas (aprox. no centro da face)."""
        from quadrature_quad import quad_geometry
        pts = np.asarray(self.points, float)
        faces = np.asarray(self.connectivity, int)
        _, nvec = quad_geometry(pts, faces, np.array([0.5]), np.array([0.5]))
        nvec = nvec[:, 0, :]
        area = np.linalg.norm(nvec, axis=1)
        return nvec / area[:, None], area


def deformed_quad_mesh(x, mesh, center):
    """Mesma deformacao radial do codigo original: p = p0 + x * (p0 - centro)."""
    pts = np.asarray(mesh.points, float).copy()
    x = np.asarray(x, float).ravel()
    N = x.size
    pts[:N] += x[:, None] * (pts[:N] - np.asarray(center, float))

    m = QuadMesh()
    m.points = list(pts)
    m.num_points = len(pts)
    m.connectivity = list(mesh.connectivity)
    m.cut_boundary = list(mesh.cut_boundary)
    m.has_nozzle = mesh.has_nozzle
    if mesh.has_nozzle:
        m.nozzle_boundary = list(mesh.nozzle_boundary)
    return m
