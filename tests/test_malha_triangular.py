"""Testes da versão 3D unificada em malha triangular (src/3D)."""

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


@pytest.mark.parametrize("nozzle", ["circle", NOZZLE])
def test_volume_inicial_igual_ao_da_gota(nozzle):
    """A calota inicial (com o topo esticado, no bico quadrado) já tem o volume pedido."""
    from optimization_3d import initial_mesh
    from quadrature_triangle import volume_3D
    for V in (0.5, 5.0, 20.0):
        mesh, c = initial_mesh(V, 6, 12, nozzle)
        assert volume_3D(np.zeros(len(mesh.free_ids)), mesh, c) == pytest.approx(V, rel=1e-10)


def test_bico_circular_prende_so_o_bico():
    """O anel do corte é a borda do bico e fica fixo: no bico circular, em z = 0
    com raio 1; no quadrado, sobre o contorno do quadrado."""
    from optimization_3d import initial_mesh
    mesh, c = initial_mesh(5.0, 6, 12, "circle")
    P = np.asarray(mesh.points)
    ring = mesh.cut_boundary
    assert mesh.nozzle_boundary == ring
    assert not set(ring) & set(mesh.free_ids)
    assert np.allclose(P[ring, 2], 0.0) and np.allclose(np.hypot(P[ring, 0], P[ring, 1]), 1.0)
    assert len(mesh.free_ids) == mesh.num_points - 12

    # bico quadrado: o anel do corte fica sobre o quadrado e também é fixo
    mesh, c = initial_mesh(5.0, 6, 12, NOZZLE)
    P = np.asarray(mesh.points)
    ring = mesh.cut_boundary
    assert mesh.nozzle_boundary == ring
    assert not set(ring) & set(mesh.free_ids)
    assert np.allclose(P[ring, 2], 0.0)
    assert np.allclose(np.max(np.abs(P[ring, :2]), axis=1), 1.0)


@pytest.mark.parametrize("nozzle", ["circle", NOZZLE])
def test_malha_final_sem_dobras(nozzle):
    """Nenhum triângulo da gota final fica virado: todos os tetraedros formados
    com um ponto interno da gota têm volume positivo, e nenhum vértice passa
    acima do plano do bico (no bico quadrado, o antigo colar dobrava)."""
    from optimization_3d import find_droplet_optimization_3d
    m = find_droplet_optimization_3d(20.0, 0.134753, 8, 16, nozzle, verbose=False)
    P = np.asarray(m.points)
    F = np.asarray(m.connectivity, dtype=int)
    o = np.array([0.0, 0.0, P[:, 2].min() / 2])
    a, b, c = P[F[:, 0]] - o, P[F[:, 1]] - o, P[F[:, 2]] - o
    assert np.all(np.einsum("ij,ij->i", a, np.cross(b, c)) > 0)
    assert P[:, 2].max() <= 1e-12


@pytest.mark.parametrize("nozzle", ["circle", NOZZLE])
@pytest.mark.parametrize("rule", ["strang3", "strang7", "strang10"])
def test_gradiente_analitico(nozzle, rule):
    """Gradientes exatos da energia e do volume conferem com diferenças centrais."""
    from optimization_3d import initial_mesh
    from quadrature_triangle import energy_3D, volume_3D, grad_energy_3D, grad_volume_3D
    mesh, c = initial_mesh(5.0, 4, 8, nozzle)
    n = len(mesh.free_ids)
    x = np.random.default_rng(2).uniform(-0.3, 0.3, n)
    e = 1e-6
    I = np.eye(n)
    fd_E = [(energy_3D(x + e * I[i], mesh, 0.3, c, rule) - energy_3D(x - e * I[i], mesh, 0.3, c, rule)) / (2 * e)
            for i in range(n)]
    fd_V = [(volume_3D(x + e * I[i], mesh, c, rule) - volume_3D(x - e * I[i], mesh, c, rule)) / (2 * e)
            for i in range(n)]
    assert np.allclose(grad_energy_3D(x, mesh, 0.3, c, rule), fd_E, rtol=1e-6, atol=1e-7)
    assert np.allclose(grad_volume_3D(x, mesh, c, rule), fd_V, rtol=1e-6, atol=1e-7)


def test_bico_circular_aproxima_o_axissimetrico():
    """Com bico circular, a gota 3D se aproxima da gota axissimétrica (EDO)
    quando a malha é refinada."""
    from optimization_3d import find_droplet_optimization_3d
    B = 0.134753
    d_edo = 1.69961   # altura da gota de volume 5 pela EDO (tiro), B = 0,134753
    erros = []
    for ns in ((6, 12), (12, 24)):
        m = find_droplet_optimization_3d(5.0, B, *ns, "circle", verbose=False)
        erros.append(abs(-np.asarray(m.points)[:, 2].min() - d_edo))
    assert erros[1] < erros[0] and erros[1] < 0.02


def test_parede_respeitada():
    """A parede x <= 1 (encostada no bico circular) é respeitada por todos os
    vértices e deixa a gota não simétrica."""
    from optimization_3d import find_droplet_optimization_3d
    m, res, _, _ = find_droplet_optimization_3d(5.0, 0.134753, 8, 16, "circle", wall=1.0,
                                                verbose=False, return_result=True)
    P = np.asarray(m.points)
    assert res.success
    assert P[:, 0].max() <= 1.0 + 1e-9
    assert P[:, 0].min() < -1.0 - 1e-3
