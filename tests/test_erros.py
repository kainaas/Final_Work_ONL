"""Testes que reproduzem os erros descritos em ERROS.md."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _fake_minimize(captured):
    """Substitui scipy.optimize.minimize: guarda as restrições e devolve x = 0."""
    def minimize(fun, x0, constraints=(), **kwargs):
        captured.append(list(constraints))
        return SimpleNamespace(x=np.zeros_like(x0), success=True, message="fake",
                               nit=0, fun=fun(np.zeros_like(x0)), jac=np.zeros_like(x0))
    return minimize


# Erro 2: volume_3D / energy_3D usam a malha original em vez da deformada
def _sphere():
    from mesh import Mesh, point
    mesh = Mesh.cut_sphere(1.0, 32, 16, 0.3, 0.0)
    return mesh, point(0.0, 0.0, 0.0), len(mesh.points)


def test_volume_3d_usa_malha_deformada():
    from quadrature_triangle import volume_3D
    mesh, c, N = _sphere()
    ratio = volume_3D(np.full(N, -0.5), mesh, c) / volume_3D(np.zeros(N), mesh, c)
    assert ratio == pytest.approx(0.125, rel=1e-10)


def test_energy_3d_usa_malha_deformada():
    from quadrature_triangle import energy_3D
    from mesh import deformed_mesh
    mesh, c, N = _sphere()
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.3, 0.3, N)
    m = deformed_mesh(x, mesh, c)
    assert energy_3D(x, mesh, 2.0, c) == pytest.approx(energy_3D(np.zeros(N), m, 2.0, c), rel=1e-12)


# Erro 3: argumento padrão mutável acumula restrições entre chamadas
def test_2d_restricoes_nao_acumulam(monkeypatch):
    import droplet_optimization as do
    captured = []
    monkeypatch.setattr(do.optm, "minimize", _fake_minimize(captured))
    do.find_droplet_optimization(1.0, 0.001, 0.1, 10)
    do.find_droplet_optimization(1.0, 0.001, 0.1, 10)
    assert [len(c) for c in captured] == [1, 1]


def test_3d_restricoes_nao_acumulam(monkeypatch):
    import optimization_3d as o3
    captured = []
    monkeypatch.setattr(o3.optm, "minimize", _fake_minimize(captured))
    nozzle = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    o3.find_droplet_optimization_3d(1.0, 0.1, 4, 8, nozzle)
    o3.find_droplet_optimization_3d(1.0, 0.1, 4, 8, nozzle)
    assert [len(c) for c in captured] == [1, 1]


# Erro 4: nomes indefinidos em run/comparison.py
def test_comparison_sem_nomes_indefinidos():
    import ast
    import builtins
    import droplet, droplet_optimization
    tree = ast.parse((ROOT / "run" / "comparison.py").read_text())
    defined = set(dir(builtins)) | set(dir(droplet)) | set(dir(droplet_optimization))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert used - defined == set()


# Erro 5: modo sessile testa o índice errado e divide por zero
def test_sessile_segmento_com_theta_constante(monkeypatch):
    import droplet as dr
    u = np.array([
        [0.1, 0.2, 0.3, 0.4],   # r
        [0.0, 0.1, 0.2, 0.3],   # z
        [0.5, 1.0, 1.0, 1.5],   # theta: constante no segmento 1
        [0.0, 0.1, 0.2, 0.3],   # V
        [0.0, 0.1, 0.2, 0.3],   # S
    ])
    monkeypatch.setattr(dr, "RK4", lambda *a, **k: dr.Result(4, 0.1, u, np.arange(4.0)))
    with np.errstate(all="raise"):
        drops = dr.find_droplets(1.0, 1.0, 1.0, 1.0, 0.1, stop_val=1.0, d_type="sessile")
    assert len(drops) == 1
    assert np.all(np.isfinite(drops[0].r))


# Erro 6: gota da otimização 2D criada com S=None
def test_2d_otimizacao_calcula_area(monkeypatch):
    import droplet_optimization as do
    monkeypatch.setattr(do.optm, "minimize", _fake_minimize([]))
    d, _ = do.find_droplet_optimization(1.0, 0.001, 0.1, 200, verbose=False)
    # x = 0: o perfil é a calota esférica inicial, de base com raio 1 e altura H;
    # a área da calota é pi * (1 + H^2)
    H = d.get_height()
    assert d.S == pytest.approx(np.pi * (1 + H**2), rel=1e-3)
    d.dimensionalize()


# Erro 7a: quadratura com número de pontos não suportado devolve 0
@pytest.mark.parametrize("method", ["integrate", "integrate_vector"])
def test_gauss_legendre_rejeita_n_invalido(method):
    from quadrature import Gauss_Legendre
    with pytest.raises(ValueError):
        getattr(Gauss_Legendre, method)(lambda x: np.ones(1), 4)


# Erro 7b: Arc.get_points_eq_space usa variável inexistente quando início == fim
def test_arco_com_inicio_igual_ao_fim():
    from objects import Arc, point
    arc = Arc(point(1.0, 0.0), point(1.0, 0.0), point(0.0, 0.0), True)
    pts = arc.get_points_eq_space(4)
    assert pts.shape == (5, 2)
    np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 1.0)
    np.testing.assert_allclose(pts[0], pts[-1], atol=1e-12)
