import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def point(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=float)


class Mesh:

    def __init__(self):
        self.points = []
        self.connectivity = []
        self.num_points = 0

        # IDs of vertices on the spherical cut boundary
        self.cut_boundary = []
        self.has_nozzle = False

    def add_point(self, p):
        self.points.append(np.asarray(p, dtype=float))
        self.num_points += 1
        return self.num_points - 1

    def add_triangle(self, i, j, k):
        self.connectivity.append((i, j, k))

    @classmethod
    def cut_sphere(
        cls,
        r: float,
        num_sectors: int,
        num_stacks: int,
        min_theta: float,
        z_center: float
    ):

        if r <= 0:
            raise ValueError("Radius must be positive.")

        if num_sectors < 3 or num_stacks < 1:
            raise ValueError("Invalid mesh resolution.")

        if not 0 < min_theta < np.pi:
            raise ValueError("min_theta must belong to (0, pi).")

        mesh = cls()

        theta_values = np.linspace(
            min_theta, np.pi, num_stacks + 1
        )

        phi_values = np.linspace(
            0, 2 * np.pi, num_sectors, endpoint=False
        )

        # Generate spherical vertices
        for theta in theta_values[:-1]:

            for phi in phi_values:

                p = point(
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta) + z_center
                )

                mesh.add_point(p)

        # Add south pole
        pole_id = mesh.add_point(point(0, 0, -r + z_center))

        # Save spherical cut boundary
        mesh.cut_boundary = list(range(num_sectors))

        # Generate triangles
        for i in range(num_stacks - 1):

            offset0 = i * num_sectors
            offset1 = (i + 1) * num_sectors

            for j in range(num_sectors):

                j1 = (j + 1) % num_sectors

                p00 = offset0 + j
                p01 = offset0 + j1
                p10 = offset1 + j
                p11 = offset1 + j1

                mesh.add_triangle(p00, p11, p01)
                mesh.add_triangle(p00, p10, p11)

        # Connect last ring to south pole
        offset = (num_stacks - 1) * num_sectors

        for j in range(num_sectors):

            j1 = (j + 1) % num_sectors

            mesh.add_triangle(
                offset + j,
                pole_id,
                offset + j1
            )

        return mesh

    @staticmethod
    def resample_contour(contour, num_points):
        """
        Resample a closed polygon using uniform arc-length spacing.
        """

        contour = np.asarray(contour, dtype=float)

        if contour.ndim != 2 or contour.shape[1] != 2:
            raise ValueError("Contour must have shape (N, 2).")

        # Remove duplicated closing vertex
        if np.allclose(contour[0], contour[-1]):
            contour = contour[:-1]

        if len(contour) < 3:
            raise ValueError("Contour must have at least 3 vertices.")

        # Signed polygon area
        x = contour[:, 0]
        y = contour[:, 1]

        area = 0.5 * np.sum(
            x * np.roll(y, -1) - np.roll(x, -1) * y
        )

        if abs(area) < 1e-14:
            raise ValueError("Contour has zero enclosed area.")

        # Ensure counterclockwise ordering
        if area < 0:
            contour = contour[::-1]

        # Close polygon
        closed = np.vstack([contour, contour[0]])

        # Segment lengths
        segments = np.diff(closed, axis=0)
        lengths = np.linalg.norm(segments, axis=1)

        if np.any(lengths < 1e-14):
            raise ValueError("Contour contains repeated consecutive points.")

        cumulative = np.concatenate([
            [0.0],
            np.cumsum(lengths)
        ])

        perimeter = cumulative[-1]

        # Uniform arc-length positions
        distances = np.linspace(
            0.0, perimeter, num_points, endpoint=False
        )

        result = np.zeros((num_points, 2))

        for i, d in enumerate(distances):

            segment = np.searchsorted(cumulative, d, side="right") - 1

            t = (d - cumulative[segment]) / lengths[segment]

            result[i] = (
                closed[segment]
                + t * segments[segment]
            )

        return result

    def attach_nozzle(
        self,
        nozzle_contour,
        z_nozzle: float,
        nozzle_length: float = 0.0
    ):
        """
        Connects the spherical cut boundary to an arbitrary
        polygonal nozzle cross-section.

        Parameters
        ----------
        nozzle_contour : array (N, 2)
            Ordered vertices of the nozzle cross-section.
        z_nozzle : float
            Height of the nozzle cross-section.
        nozzle_length : float
            Optional extrusion length of the nozzle.
        """

        if not self.cut_boundary:
            raise ValueError("No spherical cut boundary found.")

        if nozzle_length < 0:
            raise ValueError("nozzle_length must be nonnegative.")

        # Spherical boundary vertices
        sphere_ids = np.asarray(self.cut_boundary)

        sphere_ring = np.asarray(self.points)[sphere_ids]

        # Check that the nozzle is above the spherical cut
        z_cut = sphere_ring[0, 2]

        if z_nozzle <= z_cut:
            raise ValueError("Nozzle must be above the spherical cut.")

        n = len(sphere_ids)

        # Resample nozzle contour to match sphere resolution
        nozzle_ring = self.resample_contour(
            nozzle_contour, n
        )

        # Find best cyclic alignment without reversing orientation
        distances = []

        for shift in range(n):

            candidate = np.roll(nozzle_ring, shift, axis=0)

            distances.append(
                np.sum((sphere_ring[:, :2] - candidate)**2)
            )

        best_shift = np.argmin(distances)

        nozzle_ring = np.roll(
            nozzle_ring, best_shift, axis=0
        )

        # Create nozzle ring at z_nozzle
        nozzle_ids = []

        for x, y in nozzle_ring:
            nozzle_ids.append(
                self.add_point(point(x, y, z_nozzle))
            )

        # Connect sphere boundary to nozzle boundary
        for j in range(n):

            j1 = (j + 1) % n

            a = sphere_ids[j]
            b = sphere_ids[j1]

            c = nozzle_ids[j1]
            d = nozzle_ids[j]

            self.add_triangle(a, b, c)
            self.add_triangle(a, c, d)

        # Optional straight nozzle extrusion
        if nozzle_length > 0:

            top_ids = []

            for x, y in nozzle_ring:

                top_ids.append(
                    self.add_point(
                        point(x, y, z_nozzle + nozzle_length)
                    )
                )

            for j in range(n):

                j1 = (j + 1) % n

                self.add_triangle(
                    nozzle_ids[j],
                    nozzle_ids[j1],
                    top_ids[j1]
                )

                self.add_triangle(
                    nozzle_ids[j],
                    top_ids[j1],
                    top_ids[j]
                )

            self.nozzle_boundary = top_ids

        else:
            self.nozzle_boundary = nozzle_ids
        self.has_nozzle = True
        return self


    def calculate_face_normals(self):
        """
        Calculate unit normals and areas for every triangle.

        Returns
        -------
        normals : ndarray (num_triangles, 3)
            Unit normal vectors.
        areas : ndarray (num_triangles,)
            Triangle areas.
        """

        points = np.asarray(self.points, dtype=float)
        faces = np.asarray(self.connectivity, dtype=int)

        p0 = points[faces[:, 0]]
        p1 = points[faces[:, 1]]
        p2 = points[faces[:, 2]]

        cross = np.cross(p1 - p0, p2 - p0)

        double_areas = np.linalg.norm(cross, axis=1)

        if np.any(double_areas < 1e-14):
            raise ValueError("Degenerate triangle detected.")

        normals = cross / double_areas[:, None]
        areas = 0.5 * double_areas

        return normals, areas



def deformed_mesh(x: np.array, mesh: Mesh, center:np.array):
    x = np.asarray(x, dtype=float).reshape(1, -1)
    N = x.shape[1]

    m = Mesh()
    m.connectivity = mesh.connectivity.copy()
    m.cut_boundary = mesh.cut_boundary.copy()
    if mesh.has_nozzle:
        m.nozzle_boundary = mesh.nozzle_boundary.copy()
    m.points = mesh.points.copy()
    m.num_points = mesh.num_points
    
    points = np.asarray(m.points)

    points[:N, :] = (points[:N, :].T + np.vstack([x,x,x]) * (points[:N, :].T - np.asarray(center).reshape(3,1))).T

    m.points = list(points)

    return m



def plot_mesh(mesh: Mesh, show_points=False):

    points = np.asarray(mesh.points)
    connectivity = np.asarray(mesh.connectivity, dtype=int)

    # Construct triangular faces
    triangles = points[connectivity]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Surface
    surface = Poly3DCollection(
        triangles,
        facecolor="lightblue",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85
    )

    ax.add_collection3d(surface)

    # Optional vertices
    if show_points:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color="red",
            s=8
        )

    # Equal aspect ratio
    max_range = np.ptp(points, axis=0).max() / 2
    center = (points.max(axis=0) + points.min(axis=0)) / 2

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("Triangulated spherical surface")

    plt.tight_layout()
    plt.show()






def plot_normals(mesh, stride=1, scale=0.15):

    points = np.asarray(mesh.points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    triangles = points[np.asarray(mesh.connectivity)]

    surface = Poly3DCollection(
        triangles,
        facecolor="lightblue",
        edgecolor="black",
        linewidth=0.3,
        alpha=0.7
    )

    ax.add_collection3d(surface)


    normals, _ = mesh.calculate_face_normals()

    centers = triangles.mean(axis=1)

    ax.quiver(
        centers[::stride, 0],
        centers[::stride, 1],
        centers[::stride, 2],
        normals[::stride, 0],
        normals[::stride, 1],
        normals[::stride, 2],
        length=scale,
        normalize=True,
        color="red"
    )

    ax.set_box_aspect((1, 1, 1))

    ax.set_xlim(points[:, 0].min() - scale,
                points[:, 0].max() + scale)
    ax.set_ylim(points[:, 1].min() - scale,
                points[:, 1].max() + scale)
    ax.set_zlim(points[:, 2].min() - scale,
                points[:, 2].max() + scale)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Sphere
    mesh = Mesh.cut_sphere(
        r=1.0,
        num_sectors=16,
        num_stacks=8,
        min_theta=np.pi / 2.0,
        z_center= -0.05
    )


    N= len(mesh.points)
    x = np.full(N, -0.5)

    # Arbitrary nozzle contour (counterclockwise)
    nozzle = np.array([
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, 1.0]
    ])


    
    # Attach nozzle
    mesh.attach_nozzle(
        nozzle_contour=nozzle,
        z_nozzle=0.0,
        nozzle_length=0.0
    )
    plot_mesh(mesh)

    m = deformed_mesh(x, mesh, point(0.0,0.0, -0.05))
    plot_mesh(m)