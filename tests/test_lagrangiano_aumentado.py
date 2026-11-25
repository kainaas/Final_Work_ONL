"""Testes do método do lagrangiano aumentado (src/2D/augmented_lagrangian.py)."""
import warnings

import numpy as np
import pytest

SQRT2 = np.sqrt(2.0)
BOND = 0.001**2 * 1000.0 * 9.81 / 0.0728      # água, bico de 1 mm
V_MAX = 35.39486160126                        # volume máximo da gota pendente


# Problema dos Exemplos 5.2, 5.4 e 5.8 do livro (Martins & Ning):
#   minimizar x1 + 2 x2  sujeito a  x1^2/4 + x2^2 - 1 = 0  (ou <= 0)
#   solução x* = (-sqrt(2), -1/sqrt(2)), com multiplicador sqrt(2)
X_STAR = np.array([-SQRT2, -1.0 / SQRT2])


def f(x):
    return x[0] + 2.0 * x[1]


def grad_f(x):
    return np.array([1.0, 2.0])


def elipse(x):
    return x[0]**2 / 4.0 + x[1]**2 - 1.0


def jac_elipse(x):
    return np.array([[x[0] / 2.0, 2.0 * x[1]]])


def test_reproduz_exemplo_5_8_do_livro():
    """Alg. 5.2 com mu0 = 0.5 e rho = 1.1 dá os valores da Fig. 5.32 do livro."""
    from augmented_lagrangian import augmented_lagrangian
    res = augmented_lagrangian(f, [0.0, 0.0], h=elipse, jac=grad_f, jac_h=jac_elipse,
                               mu0=0.5, rho=1.1, max_iter=200)
    for k, mu, lam in [(0, "0.50", "0.000"), (2, "0.61", "1.146"), (9, "1.18", "1.413")]:
        assert f"{res.history[k]['mu']:.2f}" == mu
        assert f"{res.history[k]['lam'][0]:.3f}" == lam
    assert res.success
    np.testing.assert_allclose(res.x, X_STAR, atol=1e-7)
    assert res.lam[0] == pytest.approx(SQRT2, abs=1e-7)


def test_desigualdade_ativa():
    """Mesmo problema com g(x) <= 0 (Eqs. 5.53 e 5.54): a restrição fica ativa."""
    from augmented_lagrangian import augmented_lagrangian
    res = augmented_lagrangian(f, [0.0, 0.0], g=elipse, jac=grad_f, jac_g=jac_elipse)
    assert res.success
    np.testing.assert_allclose(res.x, X_STAR, atol=1e-7)
    assert res.sigma[0] == pytest.approx(SQRT2, abs=1e-7)


def test_desigualdade_inativa():
    """O mínimo irrestrito fica dentro da elipse: o multiplicador vai a zero."""
    from augmented_lagrangian import augmented_lagrangian
    res = augmented_lagrangian(lambda x: (x[0] - 0.5)**2 + (x[1] - 0.25)**2, [1.5, 1.0],
                               g=elipse, jac=lambda x: 2.0 * (x - [0.5, 0.25]),
                               jac_g=jac_elipse, tol=1e-10)
    assert res.success
    np.testing.assert_allclose(res.x, [0.5, 0.25], atol=1e-8)
    assert res.sigma[0] == pytest.approx(0.0, abs=1e-10)


# min |x|^2  s.a.  x1 + x2 + x3 = 1  e  0.5 - x1 <= 0
PROBLEMA_3D = dict(f=lambda x: x @ x, x0=np.zeros(3), h=lambda x: x.sum() - 1.0,
                   g=lambda x: 0.5 - x[0])


def test_igualdade_desigualdade_e_diferencas_finitas():
    """
    min |x|^2  s.a.  x1 + x2 + x3 = 1  e  0.5 - x1 <= 0.
    Solução (0.5, 0.25, 0.25), lambda = -0.5, sigma = 0.5. Sem os gradientes,
    o subproblema usa diferenças finitas e chega ao mesmo ponto.
    """
    from augmented_lagrangian import augmented_lagrangian
    exato = augmented_lagrangian(**PROBLEMA_3D, jac=lambda x: 2.0 * x,
                                 jac_h=lambda x: np.ones((1, 3)),
                                 jac_g=lambda x: np.array([[-1.0, 0.0, 0.0]]))
    dif_finitas = augmented_lagrangian(**PROBLEMA_3D)
    for res, atol in [(exato, 1e-7), (dif_finitas, 1e-5)]:
        assert res.success
        np.testing.assert_allclose(res.x, [0.5, 0.25, 0.25], atol=atol)
        assert res.lam[0] == pytest.approx(-0.5, abs=atol)
        assert res.sigma[0] == pytest.approx(0.5, abs=atol)
    assert exato.kkt < 1e-7
    assert dif_finitas.kkt is None


def test_para_quando_estagna_sem_estragar_multiplicadores():
    """
    Com diferenças finitas, tol = 1e-13 está abaixo da precisão alcançável. Sem a
    salvaguarda, mu crescia até mu_max e mu*h jogava os multiplicadores para ~1e3.
    Agora o método para, avisa (success = False) e mantém os multiplicadores bons.
    """
    from augmented_lagrangian import augmented_lagrangian
    res = augmented_lagrangian(**{**PROBLEMA_3D, "tol": 1e-13})
    assert not res.success
    assert "Stalled" in res.message
    assert res.nit < 20
    np.testing.assert_allclose(res.x, [0.5, 0.25, 0.25], atol=1e-5)
    assert res.lam[0] == pytest.approx(-0.5, abs=1e-5)
    assert res.sigma[0] == pytest.approx(0.5, abs=1e-5)


def test_limites_ficam_com_o_subproblema():
    """
    min (x1-2)^2 + (x2-2)^2  s.a.  x1 + x2 = 1  e  x1 <= 0.25 (limite, não penalizado).
    Solução (0.25, 0.75) com lambda = 2.5; a medida KKT projetada nos limites vai a zero.
    """
    from augmented_lagrangian import augmented_lagrangian
    res = augmented_lagrangian(lambda x: (x[0] - 2.0)**2 + (x[1] - 2.0)**2, [0.0, 0.0],
                               h=lambda x: x[0] + x[1] - 1.0,
                               jac=lambda x: 2.0 * (x - 2.0), jac_h=lambda x: np.ones((1, 2)),
                               bounds=[(None, 0.25), (None, None)], tol=1e-10)
    assert res.success
    np.testing.assert_allclose(res.x, [0.25, 0.75], atol=1e-8)
    assert res.lam[0] == pytest.approx(2.5, abs=1e-7)
    assert res.kkt < 1e-7


@pytest.mark.parametrize("kwargs", [{"mu0": 0.0}, {"rho": 1.0}, {"max_iter": 0}])
def test_parametros_invalidos(kwargs):
    from augmented_lagrangian import augmented_lagrangian
    with pytest.raises(ValueError):
        augmented_lagrangian(f, [0.0, 0.0], h=elipse, **kwargs)


def test_energia_e_volume_fechados_iguais_a_quadratura():
    """
    profile_energy e profile_volume dão o mesmo valor que potential_energy e
    volume_droplet, e os gradientes exatos batem com diferenças finitas.
    """
    from scipy.optimize._numdiff import approx_derivative
    from augmented_lagrangian import initial_profile, profile_energy, profile_volume
    from droplet_optimization import potential_energy, volume_droplet
    P0, c = initial_profile(10.0, 25)
    D = P0 - c.reshape(2, 1)
    x = np.random.default_rng(0).uniform(-0.3, 0.3, P0.shape[1])
    E, gE = profile_energy(P0 + x * D, BOND)
    V, gV = profile_volume(P0 + x * D)
    assert E == pytest.approx(potential_energy(x, P0, c, BOND), rel=1e-12)
    assert V == pytest.approx(volume_droplet(x, P0, c), rel=1e-12)
    fd_E = approx_derivative(lambda x: potential_energy(x, P0, c, BOND), x, method="3-point")
    fd_V = approx_derivative(lambda x: volume_droplet(x, P0, c), x, method="3-point")
    np.testing.assert_allclose(np.sum(gE * D, axis=0), fd_E, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(np.sum(gV * D, axis=0), fd_V, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("V", [5.0, V_MAX / 3.0])
def test_gota_igual_ao_slsqp(V):
    """Mesmo problema discreto: o lagrangiano aumentado chega à solução do SLSQP."""
    from augmented_lagrangian import find_droplet_augmented_lagrangian
    from droplet_optimization import find_droplet_optimization
    d_al, lam_al, res = find_droplet_augmented_lagrangian(V, 0.001, BOND, 30, verbose=False,
                                                          return_result=True)
    d_sqp, lam_sqp = find_droplet_optimization(V, 0.001, BOND, 30, verbose=False)
    assert res.success
    assert res.maxcv < 1e-10
    assert d_al.get_height() == pytest.approx(d_sqp.get_height(), abs=1e-5)
    assert lam_al == pytest.approx(lam_sqp, abs=1e-6)
    assert d_al.S > 0.0


def test_volume_maximo_fica_na_gota_pendente():
    """
    Em V*, o equilíbrio pendente é só um mínimo local. Com o mu0 padrão o método
    fica nele (igual ao SLSQP); com mu0 pequeno ele escapa para um perfil
    estrangulado, com pontos no limite x = -1, e a função avisa.
    """
    from augmented_lagrangian import find_droplet_augmented_lagrangian
    from droplet_optimization import find_droplet_optimization
    d_sqp, lam_sqp = find_droplet_optimization(V_MAX, 0.001, BOND, 30, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        d_al, lam_al = find_droplet_augmented_lagrangian(V_MAX, 0.001, BOND, 30, verbose=False)
    assert d_al.get_height() == pytest.approx(d_sqp.get_height(), abs=1e-4)
    assert lam_al == pytest.approx(lam_sqp, abs=1e-5)
    with pytest.warns(RuntimeWarning, match="collapsed"):
        find_droplet_augmented_lagrangian(V_MAX, 0.001, BOND, 30, mu0=10.0, verbose=False)


def test_lambda0_usa_a_convencao_do_retorno():
    """
    Partindo do multiplicador já conhecido (na convenção do valor devolvido), o
    primeiro subproblema já sai quase viável; partindo de zero, o erro de volume é
    cerca de lambda V0 / mu0. Com o sinal trocado, o erro dobraria.
    """
    from augmented_lagrangian import find_droplet_augmented_lagrangian
    _, lam, frio = find_droplet_augmented_lagrangian(5.0, 0.001, BOND, 30, verbose=False,
                                                     return_result=True)
    _, lam_quente, quente = find_droplet_augmented_lagrangian(5.0, 0.001, BOND, 30, lambda0=lam,
                                                              verbose=False, return_result=True)
    assert frio.history[0]["violation"] == pytest.approx(lam * 5.0 / 1000.0, rel=0.1)
    assert quente.history[0]["violation"] < 1e-2 * frio.history[0]["violation"]
    assert lam_quente == pytest.approx(lam, abs=1e-6)


def test_diferencas_finitas_igual_ao_gradiente_exato():
    """analytic_gradient=False usa as funções do SLSQP com diferenças finitas."""
    from augmented_lagrangian import find_droplet_augmented_lagrangian
    d1, l1 = find_droplet_augmented_lagrangian(5.0, 0.001, BOND, 15, verbose=False)
    d2, l2 = find_droplet_augmented_lagrangian(5.0, 0.001, BOND, 15, analytic_gradient=False,
                                               verbose=False)
    assert d2.get_height() == pytest.approx(d1.get_height(), abs=1e-4)
    assert l2 == pytest.approx(l1, abs=1e-5)


def test_slsqp_2d_comeca_viavel_e_fica_perto_do_volume():
    """A calota inicial 2D já tem o volume pedido, e o SLSQP (com a energia
    escalada) não se afasta do volume durante as iterações: sem a escala, o
    volume 20 pulava para 35 no segundo passo."""
    from droplet_optimization import find_droplet_optimization, initial_profile, profile_volume
    for V in (5.0, 20.0):
        P0, _ = initial_profile(V, 30)
        assert profile_volume(P0)[0] == pytest.approx(V, rel=1e-12)
        vols = []
        find_droplet_optimization(V, 0.001, BOND, 30, verbose=False,
                                  callback=lambda P: vols.append(profile_volume(P)[0]))
        assert max(abs(v - V) for v in vols) < 0.01 * V
        assert vols[-1] == pytest.approx(V, rel=1e-8)
