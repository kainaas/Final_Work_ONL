"""
Gera os dados das animações do vídeo (video/dados/*.npz).

Uso, de dentro da pasta video/:

    python gerar_dados.py

As cenas do Manim (cenas.py) só leem esses arquivos; nenhuma otimização roda
durante o render.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import scipy.optimize as optm

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "src" / "2D"))
sys.path.insert(0, str(RAIZ / "src" / "3D"))

import const                                                    # noqa: E402
from droplet import find_droplets, shooting                     # noqa: E402
from droplet_optimization import find_droplet_optimization, initial_profile  # noqa: E402
from augmented_lagrangian import (find_droplet_augmented_lagrangian,  # noqa: E402
                                  profile_energy, profile_volume)
from mesh import deformed_mesh                                  # noqa: E402
from optimization_3d import find_droplet_optimization_3d        # noqa: E402

DADOS = AQUI / "dados"

GAMMA, RHO, A_BICO = 0.0728, 1000.0, 0.001
BOND = RHO * const.g * A_BICO**2 / GAMMA
A_STAR, V_STAR = 1.19721119513, 35.39486160126
MAX_S, DS = 5.0 * np.pi, 0.001
V_DEMO = 20.0                     # volume das cenas de otimização
MALHA_3D = (12, 24)
QUADRADO = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])


def calado(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def partes_da_energia(P):
    """(superfície, gravidade, total) da energia adimensional de um perfil (2, N)."""
    r1, z1, r2, z2 = P[0, :-1], P[1, :-1], P[0, 1:], P[1, 1:]
    sup = np.pi * np.sum(np.hypot(r2 - r1, z2 - z1) * (r1 + r2))
    total = np.pi * profile_energy(P, BOND)[0]
    return sup, total - sup, total


def gota_tiro(V):
    """Perfil (2, N), do bico ao ápice, da gota de volume V pela EDO."""
    if V >= 35.3:
        A = A_STAR
    elif V > 34.0:
        # perto da dobra da curva V x A a secante diverge (falhava em V = 35);
        # no ramo da gota pendente, V cai de V* a ~30 quando A vai de A* a 1,25
        def falta(A):
            return max(g.V for g in calado(find_droplets, A, BOND, A_BICO, MAX_S, DS)) - V
        A = optm.brentq(falta, A_STAR + 1e-9, 1.25, xtol=1e-13)
    elif V > 20.0:
        A = shooting(V, 1.2, 1.4, BOND, A_BICO)
    elif V > 10.0:
        A = shooting(V, 1.3, 1.6, BOND, A_BICO)
    elif V > 5.0:
        A = shooting(V, 1.5, 1.9, BOND, A_BICO)
    elif V > 4.5:
        A = shooting(V, 1.2, 1.9, BOND, A_BICO)
    elif V > 2.17221828749:
        A = shooting(V, 1.9, 2.04, BOND, A_BICO)
    else:
        A = shooting(V, 0.0, 2.0, BOND, A_BICO)
    gotas = calado(find_droplets, A, BOND, A_BICO, MAX_S, DS)
    g = max(gotas, key=lambda g: g.V) if V >= 35.3 else min(gotas[:2], key=lambda g: abs(g.V - V))
    r, z = np.asarray(g.r), np.asarray(g.z)
    return reamostra(np.vstack([np.flip(r), np.flip(z) - z[-1]]))


def reamostra(P, n=200):
    """Reamostra o perfil (2, N) em n pontos igualmente espaçados ao longo dele."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=1)))])
    t = np.linspace(0.0, s[-1], n)
    return np.vstack([np.interp(t, s, P[0]), np.interp(t, s, P[1])])


def otimizacao_2d():
    """Perfis a cada iteração do SLSQP (2D, N = 30); o primeiro é a calota inicial."""
    hist = [initial_profile(V_DEMO, 30)[0]]
    calado(find_droplet_optimization, V_DEMO, A_BICO, BOND, 30, verbose=False,
           callback=lambda P: hist.append(np.array(P)))
    perfis = np.stack(hist)
    energias = np.array([partes_da_energia(P) for P in perfis])
    volumes = np.array([profile_volume(P)[0] for P in perfis])
    np.savez(DADOS / "otimizacao_2d.npz", perfis=perfis, energias=energias, volumes=volumes,
             edo=gota_tiro(V_DEMO), V=V_DEMO)


def cabo_de_guerra():
    """Família de formas com o mesmo volume, da mais achatada à mais esticada.

    Partimos da gota de equilíbrio (EDO) e a esticamos na vertical por um fator
    s. Para voltar ao volume certo, todos os pontos menos o do bico se afastam
    (ou se aproximam) do eixo da gota na mesma proporção, como na
    parametrização da otimização. A forma de menor energia total da família
    fica perto de s = 1, a gota de equilíbrio.
    """
    P_otimo = gota_tiro(V_DEMO)
    fatores = np.linspace(0.6, 1.6, 61)
    formas, partes = [], []
    for s in fatores:
        P = P_otimo.copy()
        P[1] *= s
        c = np.array([[0.0], [P[1].mean()]])

        def corrigido(a, P=P, c=c):
            Q = P.copy()
            Q[:, 1:] = c + (1.0 + a) * (P[:, 1:] - c)
            return Q
        a = optm.brentq(lambda a: profile_volume(corrigido(a))[0] - V_DEMO, -0.9, 2.0)
        P = corrigido(a)
        formas.append(P)
        partes.append(partes_da_energia(P))
    np.savez(DADOS / "cabo_de_guerra.npz", formas=np.stack(formas), partes=np.array(partes),
             fatores=fatores)


def crescendo():
    """Gotas de volume crescente até o volume máximo, e a gota que se solta."""
    volumes = np.concatenate([np.linspace(1.0, 34.0, 23), [34.5, 35.0, V_STAR]])
    ok, perfis = [], []
    for V in volumes:
        try:
            perfis.append(gota_tiro(V))
            ok.append(V)
        except RuntimeError as erro:
            print("   tiro falhou para V =", V, ":", erro)
    volumes, perfis = np.array(ok), np.stack(perfis)
    g, _, _ = calado(find_droplet_augmented_lagrangian, V_STAR, A_BICO, BOND, 30, mu0=10.0,
                     verbose=False, return_result=True)
    r, z = np.asarray(g.r), np.asarray(g.z)
    solta = np.vstack([np.flip(r), np.flip(z) - z[-1]])
    np.savez(DADOS / "crescendo.npz", volumes=volumes, perfis=perfis, solta=solta)


def trilha():
    """Exemplo 5.8 do livro com o SLSQP: os passos até o ótimo."""
    it = [np.array([2.0, 0.5])]
    elipse = lambda x: x[0]**2 / 4.0 + x[1]**2 - 1.0                        # noqa: E731
    optm.minimize(lambda x: x[0] + 2.0 * x[1], it[0], jac=lambda x: np.array([1.0, 2.0]),
                  method="SLSQP", callback=lambda xk: it.append(np.array(xk)),
                  constraints=[optm.NonlinearConstraint(
                      elipse, 0.0, 0.0, jac=lambda x: np.array([[x[0] / 2.0, 2.0 * x[1]]]))],
                  options={"ftol": 1e-12})
    np.savez(DADOS / "trilha.npz", iteracoes=np.stack(it))


def gotas_3d():
    """Malhas 3D: iterações da gota circular, gota no bico quadrado e com parede."""
    hist = []
    m, res, m0, c = find_droplet_optimization_3d(
        V_DEMO, BOND, *MALHA_3D, "circle", verbose=False, return_result=True,
        callback=lambda xk: hist.append(np.array(xk)))
    quadros = [np.asarray(m0.points)] + [np.asarray(deformed_mesh(x, m0, c).points) for x in hist]
    np.savez(DADOS / "gota3d_iteracoes.npz", points=np.stack(quadros),
             connectivity=np.asarray(m0.connectivity, dtype=int), edo=gota_tiro(V_DEMO))
    for nome, bico, parede in (("quadrado", QUADRADO, None), ("parede", "circle", 1.0)):
        m = find_droplet_optimization_3d(V_DEMO, BOND, *MALHA_3D, bico, wall=parede, verbose=False)
        np.savez(DADOS / f"gota3d_{nome}.npz", points=np.asarray(m.points),
                 connectivity=np.asarray(m.connectivity, dtype=int))


if __name__ == "__main__":
    DADOS.mkdir(exist_ok=True)
    for passo in (trilha, otimizacao_2d, cabo_de_guerra, crescendo, gotas_3d):
        print("gerando", passo.__name__)
        passo()
    print("dados em", DADOS)
