"""Testes da versão 3D unificada em malha triangular (src/3D)."""
from types import SimpleNamespace

import numpy as np
import pytest

NOZZLE = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])


def _mesh_com_bico():
    from mesh import Mesh, point
    center = point(0.0, 0.0, -0.2)
    mesh = Mesh.cut_sphere(1.0, 12, 6, np.pi / 2.0, center[2])
    N = len(mesh.points)
    mesh.attach_nozzle(NOZZLE, 0.0, 0.0)
    return mesh, center, N


def _referencia_laco(x, mesh, Bond, center, rule):
    """Implementação original (um triângulo por vez), usada como gabarito."""
    from mesh import deformed_mesh
    from quadrature_triangle import Quadrature
    m = deformed_mesh(x, mesh, center)
    normals, areas = m.calculate_face_normals()
    tri = np.asarray(m.points)[np.asarray(m.connectivity, dtype=int)]
    E = V = 0.0
    for (a, b, c), n, A in zip(tri, normals, areas):
        T = lambda p: p[0] * (b - a) + p[1] * (c - a) + a
        E += Quadrature.integrate(lambda p: 2 * A * (1 + Bond / 2 * n[2] * T(p)[2] ** 2), rule)
        V += Quadrature.integrate(lambda p: 2 * A * np.dot(T(p), n), rule)
    return E, V / 3.0


@pytest.mark.parametrize("rule", ["strang3", "strang7", "strang10"])
def test_vetorizado_igual_ao_laco(rule):
    from quadrature_triangle import energy_3D, volume_3D
    mesh, c, N = _mesh_com_bico()
    x = np.random.default_rng(1).uniform(-0.3, 0.3, N)
    E_ref, V_ref = _referencia_laco(x, mesh, 2.0, c, rule)
    assert energy_3D(x, mesh, 2.0, c, rule) == pytest.approx(E_ref, rel=1e-12)
    assert volume_3D(x, mesh, c, rule) == pytest.approx(V_ref, rel=1e-12)


def test_regra_desconhecida():
    from quadrature_triangle import energy_3D
    mesh, c, N = _mesh_com_bico()
    with pytest.raises(ValueError):
        energy_3D(np.zeros(N), mesh, 1.0, c, "gauss99")


def _captura_minimize(captured):
    def minimize(fun, x0, constraints=(), bounds=None, **kwargs):
        captured.append(bounds)
        return SimpleNamespace(x=np.zeros_like(x0), success=True, message="fake",
                               nit=0, fun=fun(np.zeros_like(x0)))
    return minimize


@pytest.mark.parametrize("fix", [False, True])
def test_fix_boundary_prende_linha_de_contato(monkeypatch, fix):
    import optimization_3d as o3
    captured = []
    monkeypatch.setattr(o3.optm, "minimize", _captura_minimize(captured))
    o3.find_droplet_optimization_3d(1.0, 0.1, 4, 8, NOZZLE, fix_boundary=fix, verbose=False)
    b = captured[0]
    ring = list(range(8))  # cut_sphere numera o anel do corte primeiro
    if fix:
        assert np.all(b.lb[ring] == 0.0) and np.all(b.ub[ring] == 0.0)
    else:
        assert np.all(b.lb[ring] < 0.0) and np.all(np.isinf(b.ub[ring]))
