"""
Gera as figuras (figuras/*.pdf), as tabelas (tabelas/*.tex) e os números
(dados/resultados.json) usados em relatorio.tex.

Uso, de dentro da pasta relatorio/:

    python gerar_figuras.py               # reaproveita dados/resultados.json, se existir
    python gerar_figuras.py --recalcular  # refaz todas as contas (alguns minutos)

Depois de mudar o código, rode com --recalcular: os dados antigos não têm as
chaves novas (gota3d circular/quadrado/parede, eixo3d, exemplo_livro_slsqp).
"""
import argparse
import contextlib
import io
import json
import platform
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy
import scipy.optimize

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "src" / "2D"))
sys.path.insert(0, str(RAIZ / "src" / "3D"))

import const                                                    # noqa: E402
from droplet import find_droplets, shooting                     # noqa: E402
from droplet_optimization import find_droplet_optimization      # noqa: E402
from augmented_lagrangian import (augmented_lagrangian,         # noqa: E402
                                  find_droplet_augmented_lagrangian,
                                  profile_energy)
from optimization_3d import find_droplet_optimization_3d        # noqa: E402

FIGURAS = AQUI / "figuras"
TABELAS = AQUI / "tabelas"
DADOS = AQUI / "dados"

# água com bico de raio a = 1 mm (mesmos valores de run/pendant.py)
GAMMA = 0.0728                    # N/m
RHO = 1000.0                      # kg/m^3
A_BICO = 0.001                    # m
BOND = RHO * const.g * A_BICO**2 / GAMMA
A_STAR = 1.19721119513            # pressão no ápice da gota de volume máximo
V_STAR = 35.39486160126           # volume máximo da gota pendente
MAX_S, DS = 5.0 * np.pi, 0.001    # integração da EDO (como em run/pendant.py)

# versão 3D: mesmo B do 2D (o comprimento de referência é o raio do bico
# circular, a = 1 mm); o bico quadrado é [-1, 1]^2 nessas unidades
BICO_3D = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
BOND_3D = BOND
V_3D = 20.0                       # gotas 3D da Figura gota3d (circular, quadrado, parede)
MALHA_3D = (12, 24)               # anéis x setores
PAREDE = 1.0                      # parede x <= 1, encostada no bico circular
V_EIXO = (5.0, 20.0)              # volumes da comparação 3D x axissimétrico
MALHAS_EIXO = ((6, 12), (8, 16), (12, 24), (16, 32), (24, 48))


# ---------------------------------------------------------------------------
# contas
# ---------------------------------------------------------------------------

def calado(f, *args, **kwargs):
    """Chama f sem mostrar o que ela imprime; devolve (resultado, texto impresso)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f(*args, **kwargs)
    return res, buf.getvalue()


def gota_tiro(V):
    """Gota pendente de volume V pelo método do tiro (intervalos de run/comparison.py)."""
    if V >= 35.3:
        A = A_STAR
    elif V > 34.0:
        A = shooting(V, 1.18, 1.20, BOND, A_BICO)
    elif V > 20.0:
        A = shooting(V, 1.2, 1.4, BOND, A_BICO)
    elif V > 10.0:
        A = shooting(V, 1.3, 1.6, BOND, A_BICO)
    elif V > 5.0:
        A = shooting(V, 1.5, 1.9, BOND, A_BICO)
    elif V > 4.5:
        A = shooting(V, 1.2, 1.9, BOND, A_BICO)
    elif V > 2.17221828749:
        # logo acima da pressão máxima o intervalo de run/comparison.py, (1.2, 1.9),
        # leva a secante para fora do ramo certo; este começa perto de A_max
        A = shooting(V, 1.9, 2.04, BOND, A_BICO)
    else:
        A = shooting(V, 0.0, 2.0, BOND, A_BICO)
    gotas = find_droplets(A, BOND, A_BICO, MAX_S, DS)
    if V >= 35.3:
        gota = max(gotas, key=lambda g: g.V)
    else:
        gota = min(gotas[:2], key=lambda g: abs(g.V - V))
    d = gota.get_height()
    return {"A": float(A), "V": float(gota.V), "d": float(d),
            "lam": float((A - BOND * d) / np.pi),
            "r": gota.r.tolist(), "z": gota.z.tolist()}


def gota_slsqp(V, N):
    t = time.perf_counter()
    (gota, lam), saida = calado(find_droplet_optimization, V, A_BICO, BOND, N, verbose=False)
    dt = time.perf_counter() - t
    it = re.search(r"Iterations:\s*(\d+)", saida)
    return {"d": float(gota.get_height()), "lam": float(lam), "tempo": dt,
            "iteracoes": int(it.group(1)) if it else None,
            "r": gota.r.tolist(), "z": gota.z.tolist()}


def gota_al(V, N, **kwargs):
    t = time.perf_counter()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        gota, lam, res = find_droplet_augmented_lagrangian(V, A_BICO, BOND, N, verbose=False,
                                                           return_result=True, **kwargs)
    dt = time.perf_counter() - t
    return {"d": float(gota.get_height()), "lam": float(lam), "tempo": dt,
            "externas": int(res.nit),
            "internas": int(sum(h["inner_nit"] for h in res.history)),
            "nfev": int(res.nfev), "violacao": float(res.maxcv),
            "kkt": None if res.kkt is None else float(res.kkt),
            "sucesso": bool(res.success), "mensagem": res.message,
            "avisos": [str(w.message) for w in avisos],
            "r": gota.r.tolist(), "z": gota.z.tolist()}


def gota_3d(V, malha, bico3d, parede, analytic_gradient=True):
    """Gota 3D pelo SLSQP; devolve (resumo, malha final)."""
    t = time.perf_counter()
    m, res, _, _ = find_droplet_optimization_3d(V, BOND_3D, *malha, bico3d, wall=parede,
                                                analytic_gradient=analytic_gradient,
                                                verbose=False, return_result=True)
    dt = time.perf_counter() - t
    P = np.asarray(m.points, dtype=float)
    mult = np.ravel(res.multipliers)
    resumo = {"V": V, "malha": list(malha), "n_vertices": int(len(P)),
              "n_variaveis": int(len(res.x)), "n_triangulos": int(len(m.connectivity)),
              "iteracoes": int(res.nit), "tempo": dt, "sucesso": bool(res.success),
              "energia": float(res.fun), "altura": float(-P[:, 2].min()),
              # na convenção do relatório (energia dividida por pi)
              "lam": float(mult[0] / np.pi),
              "x_max": float(P[:, 0].max()), "x_min": float(P[:, 0].min())}
    if parede is not None:
        # multiplicadores das restrições da parede (convenção do SciPy: c(x) >= 0,
        # sigma >= 0); ativas são as de multiplicador não nulo
        sigma = mult[1:]
        resumo["ativas"] = int(np.sum(np.abs(sigma) > 1e-8))
        resumo["n_restricoes_parede"] = int(len(sigma))
        resumo["sigma_max"] = float(sigma.max())
        resumo["sigma_min"] = float(sigma.min())
        resumo["sigma_soma"] = float(sigma.sum())
    return resumo, m


def pontos_do_perfil(r, z):
    """Reconstrói os pontos (bico -> ápice) a partir do perfil (r, z) de uma Droplet."""
    r, z = np.asarray(r), np.asarray(z)
    return np.vstack([np.flip(r), np.flip(z) - z[-1]])


def calcular_lambda_V():
    """Multiplicador pela EDO e pelo lagrangiano aumentado (N = 64) para vários volumes."""
    lv = []
    for V in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0,
              30.0, 33.0, 34.5, V_STAR):
        try:
            t = gota_tiro(V)
        except RuntimeError as erro:
            print("   tiro falhou para V =", V, ":", erro)
            continue
        al = gota_al(V, 64)
        lv.append({"V": V, "A": t["A"], "d_tiro": t["d"], "lam_tiro": t["lam"],
                   "d_al": al["d"], "lam_al": al["lam"]})
    return lv


def calcular():
    dados = {"ambiente": {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
             "bond": BOND, "bond_3d": BOND_3D, "A_star": A_STAR, "V_star": V_STAR}

    print("1/9 família de gotas pelo método do tiro")
    dados["familia"] = [gota_tiro(V) for V in (1.0, 5.0, 10.0, 20.0, 30.0, V_STAR)]

    print("2/9 curva V x A, volume máximo e pressão máxima")
    curva = []
    for A in np.linspace(0.02, 2.2, 219):
        for i, g in enumerate(find_droplets(A, BOND, A_BICO, MAX_S, 0.002)):
            curva.append([float(A), float(g.V), float(g.get_height()), i])
    dados["curva_VA"] = curva
    vizinhos = {}
    for dA in (-1e-4, 1e-4):
        vizinhos[str(dA)] = max(g.V for g in find_droplets(A_STAR + dA, BOND, A_BICO, MAX_S, DS))
    dados["vizinhanca_A_star"] = vizinhos

    # pressão máxima: o maior A em que ainda existe o ramo das gotas pequenas
    # (as duas primeiras interseções se fundem ali; acima dele só há gotas com V > 5,
    # vindas de cruzamentos posteriores do perfil, como em run/pendant.py)
    def ramo_pequeno(A):
        gotas = find_droplets(A, BOND, A_BICO, MAX_S, DS)
        return len(gotas) > 0 and min(g.V for g in gotas) < 5.0
    lo, hi = 2.0, 2.1
    assert ramo_pequeno(lo) and not ramo_pequeno(hi)
    while hi - lo > 1e-10:
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if ramo_pequeno(m) else (lo, m)
    gotas = sorted(find_droplets(lo, BOND, A_BICO, MAX_S, DS), key=lambda g: g.V)
    dados["pressao_maxima"] = {"A": lo, "V": float(gotas[0].V), "V2": float(gotas[1].V),
                               "d": float(gotas[0].get_height())}

    print("3/9 comparação EDO x SLSQP x lagrangiano aumentado (N = 30)")
    dados["comparacao"] = []
    for nome, V in (("V*/3", V_STAR / 3), ("2V*/3", 2 * V_STAR / 3), ("V*", V_STAR)):
        dados["comparacao"].append({"nome": nome, "V": V, "tiro": gota_tiro(V),
                                    "slsqp": gota_slsqp(V, 30), "al": gota_al(V, 30)})

    print("4/9 convergência na malha (V = 2V*/3)")
    V = 2 * V_STAR / 3
    conv = {"V": V, "N": [8, 16, 32, 64, 128], "slsqp": [], "al": []}
    for N in conv["N"]:
        for metodo, f in (("slsqp", gota_slsqp), ("al", gota_al)):
            res = f(V, N)
            res.pop("r"), res.pop("z")
            conv[metodo].append(res)
    dados["convergencia"] = conv

    print("5/9 histórico do lagrangiano aumentado (V = V*/3, N = 30)")
    V = V_STAR / 3
    _, _, res = find_droplet_augmented_lagrangian(V, A_BICO, BOND, 30, verbose=False,
                                                  return_result=True)
    depois = [h["lam"] for h in res.history[1:]] + [res.lam]
    dados["historico"] = [{"k": h["k"], "mu": h["mu"], "violacao": h["violation"],
                           "kkt": h["kkt"], "internas": int(h["inner_nit"]),
                           "lam": float(-lam[0] / V)}
                          for h, lam in zip(res.history, depois)]

    print("6/9 multiplicador em função do volume")
    dados["lambda_V"] = calcular_lambda_V()

    print("7/9 volume máximo: mu0 = 1000 x mu0 = 10")
    desc = {}
    for nome, mu0 in (("mu1000", 1000.0), ("mu10", 10.0)):
        g = gota_al(V_STAR, 30, mu0=mu0)
        g["energia"] = float(np.pi * profile_energy(pontos_do_perfil(g["r"], g["z"]), BOND)[0])
        desc[nome] = g
    dados["descolamento"] = desc

    print("8/9 gotas 3D: bico circular, bico quadrado e parede")
    dados["gota3d"] = {}
    for nome, bico3d, parede in (("circular", "circle", None), ("quadrado", BICO_3D, None),
                                 ("parede", "circle", PAREDE)):
        g, malha = gota_3d(V_3D, MALHA_3D, bico3d, parede)
        np.savez(DADOS / f"gota3d_{nome}.npz", points=np.asarray(malha.points, dtype=float),
                 connectivity=np.asarray(malha.connectivity, dtype=int))
        dados["gota3d"][nome] = g
    # custo do gradiente por diferenças finitas (mesma gota circular, malha 8 x 16)
    dados["gota3d_gradiente"] = {
        "exato": gota_3d(V_3D, (8, 16), "circle", None)[0],
        "diferencas_finitas": gota_3d(V_3D, (8, 16), "circle", None, analytic_gradient=False)[0]}

    print("9/9 bico circular x axissimétrico (refinamento da malha)")
    eixo = []
    for V in V_EIXO:
        t = gota_tiro(V)
        t["energia"] = float(np.pi * profile_energy(pontos_do_perfil(t["r"], t["z"]), BOND)[0])
        malhas = []
        for ns in MALHAS_EIXO:
            g, malha = gota_3d(V, ns, "circle", None)
            P = np.asarray(malha.points, dtype=float)
            g["r_vertices"] = np.hypot(P[:, 0], P[:, 1]).tolist()
            g["z_vertices"] = P[:, 2].tolist()
            malhas.append(g)
        eixo.append({"V": V, "tiro": t, "malhas": malhas})
    dados["eixo3d"] = eixo

    print("exemplo 5.8 do livro")
    elipse = lambda x: x[0]**2 / 4.0 + x[1]**2 - 1.0                      # noqa: E731
    res = augmented_lagrangian(lambda x: x[0] + 2.0 * x[1], [0.0, 0.0], h=elipse,
                               jac=lambda x: np.array([1.0, 2.0]),
                               jac_h=lambda x: np.array([[x[0] / 2.0, 2.0 * x[1]]]),
                               mu0=0.5, rho=1.1, max_iter=200)
    dados["exemplo_livro"] = {"historico": [{"k": h["k"], "mu": h["mu"], "lam": float(h["lam"][0]),
                                             "violacao": h["violation"]} for h in res.history],
                              "x": res.x.tolist(), "lam": float(res.lam[0]),
                              "nit": int(res.nit), "sucesso": bool(res.success)}
    print("exemplo 5.8 do livro com o SLSQP")
    it = [np.array([2.0, 0.5])]
    res = scipy.optimize.minimize(
        lambda x: x[0] + 2.0 * x[1], it[0], jac=lambda x: np.array([1.0, 2.0]), method="SLSQP",
        constraints=[scipy.optimize.NonlinearConstraint(elipse, 0.0, 0.0,
                                                         jac=lambda x: np.array([[x[0] / 2.0, 2.0 * x[1]]]))],
        callback=lambda xk: it.append(np.array(xk)), options={"ftol": 1e-12})
    dados["exemplo_livro_slsqp"] = {"iteracoes": [x.tolist() for x in it], "x": res.x.tolist(),
                                    "lam": float(np.ravel(res.multipliers)[0]), "nit": int(res.nit),
                                    "sucesso": bool(res.success)}
    return dados


# ---------------------------------------------------------------------------
# figuras
# ---------------------------------------------------------------------------

COR = {"tiro": "black", "slsqp": "tab:blue", "al": "tab:red"}
ROTULO = {"tiro": "EDO (tiro)", "slsqp": "SLSQP", "al": "Lagrangiano aumentado"}


def perfil(r, z):
    """Perfil completo (as duas metades), com o bico em z = 0 e a gota para baixo."""
    r, z = np.asarray(r), np.asarray(z)
    d = z[-1]
    return np.hstack([-np.flip(r), r]), np.hstack([np.flip(z), z]) - d


def bico(ax, altura=0.45):
    for x0 in (1.0, -1.25):
        ax.add_patch(plt.Rectangle((x0, 0.0), 0.25, altura, color="0.7", lw=0, zorder=3))


def salvar(fig, nome):
    fig.savefig(FIGURAS / nome)
    plt.close(fig)


def figura_familia(dados):
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    cores = plt.cm.viridis(np.linspace(0.0, 0.9, len(dados["familia"])))
    for g, c in zip(dados["familia"], cores):
        x, z = perfil(g["r"], g["z"])
        rot = rf"$\hat V = {g['V']:.1f}$" if abs(g["V"] - V_STAR) > 1e-6 else rf"$\hat V^* = {g['V']:.2f}$"
        ax.plot(x, z, color=c, lw=1.6, label=rot)
    bico(ax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\hat r$")
    ax.set_ylabel(r"$\hat z$")
    ax.legend(loc="lower right")
    salvar(fig, "familia_tiro.pdf")


def figura_curva_VA(dados):
    c = np.asarray(dados["curva_VA"])
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for i, m in zip(range(int(c[:, 3].max()) + 1), ("o", "s", "^", "v", "D")):
        sel = c[:, 3] == i
        ax.plot(c[sel, 1], c[sel, 0], m, ms=2.5, ls="none", label=f"{i + 1}ª interseção com $\\hat r = 1$")
    pm = dados["pressao_maxima"]
    ax.plot(V_STAR, A_STAR, "k*", ms=11)
    ax.annotate(rf"$\hat V^* = {V_STAR:.2f}$" + "\n" + rf"$A = {A_STAR:.4f}$", (V_STAR, A_STAR),
                xytext=(24.0, 0.45), textcoords="data", arrowprops={"arrowstyle": "->"})
    ax.plot(pm["V"], pm["A"], "k*", ms=11)
    ax.annotate(rf"$A_{{\max}} = {pm['A']:.4f}$" + "\n" + rf"$\hat V = {pm['V']:.3f}$", (pm["V"], pm["A"]),
                xytext=(7.0, 0.75), textcoords="data", arrowprops={"arrowstyle": "->"})
    ax.set_xlim(0, 40)
    ax.set_xlabel(r"$\hat V$")
    ax.set_ylabel(r"$A$ (pressão adimensional no ápice)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7.5)
    salvar(fig, "curva_VA.pdf")


def figura_comparacao(dados):
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 4.6))
    titulos = {"V*/3": r"$\hat V = \hat V^*/3$", "2V*/3": r"$\hat V = 2\hat V^*/3$", "V*": r"$\hat V = \hat V^*$"}
    for i, (ax, caso) in enumerate(zip(axs, dados["comparacao"])):
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-5.7, 0.7)
        if i > 0:
            ax.tick_params(labelleft=False)
        x, z = perfil(caso["tiro"]["r"], caso["tiro"]["z"])
        ax.plot(x, z, "-", color=COR["tiro"], lw=1.2, label=ROTULO["tiro"])
        for metodo, marca in (("slsqp", "o"), ("al", "x")):
            x, z = perfil(caso[metodo]["r"], caso[metodo]["z"])
            ax.plot(x, z, marca, color=COR[metodo], ms=4, mfc="none", label=ROTULO[metodo] + " ($N = 30$)")
        bico(ax)
        ax.set_aspect("equal")
        ax.set_title(titulos[caso["nome"]])
        ax.set_xlabel(r"$\hat r$")
    axs[0].set_ylabel(r"$\hat z$")
    axs[0].legend(loc="lower left", fontsize=7)
    salvar(fig, "comparacao_perfis.pdf")


def figura_convergencia(dados):
    conv = dados["convergencia"]
    ref = next(c for c in dados["comparacao"] if c["nome"] == "2V*/3")["tiro"]
    N = np.asarray(conv["N"], dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for metodo, marca, ls in (("slsqp", "o", "-"), ("al", "x", "--")):
        ed = [abs(c["d"] - ref["d"]) for c in conv[metodo]]
        el = [abs(c["lam"] - ref["lam"]) for c in conv[metodo]]
        ax1.loglog(N, ed, marca + ls, color=COR[metodo], mfc="none", label=rf"$|\hat d - \hat d_{{EDO}}|$, {ROTULO[metodo]}")
        ax1.loglog(N, el, marca + ls, color=COR[metodo], mfc="none", alpha=0.5,
                   label=rf"$|\hat\lambda - \hat\lambda_{{EDO}}|$, {ROTULO[metodo]}")
        ax2.loglog(N, [c["tempo"] for c in conv[metodo]], marca + ls, color=COR[metodo],
                   mfc="none", label=ROTULO[metodo])
    e0 = abs(conv["slsqp"][0]["d"] - ref["d"])
    ax1.loglog(N, e0 * (N / N[0])**-2, ":", color="0.4", label=r"inclinação $N^{-2}$")
    fig.subplots_adjust(wspace=0.3)
    ax1.set_xlabel("número de pontos $N$")
    ax1.set_ylabel("erro em relação à EDO")
    ax1.legend(fontsize=6.5, loc="lower left")
    ax2.set_xlabel("número de pontos $N$")
    ax2.set_ylabel("tempo (s)")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.set_xticks(N)
        ax.set_xticklabels([str(int(n)) for n in N])
        ax.minorticks_off()
    salvar(fig, "convergencia.pdf")


def figura_historico(dados):
    h = dados["historico"]
    ref = next(c for c in dados["comparacao"] if c["nome"] == "V*/3")
    k = [i["k"] for i in h]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax1.semilogy(k, [i["violacao"] for i in h], "o-", color=COR["al"], label=r"$|h(x_k)|$")
    ax1.semilogy(k, [i["kkt"] for i in h], "s--", color="0.4", mfc="none",
                 label="KKT (gradiente projetado)")
    for i in h:
        ax1.annotate(rf"$\mu = {i['mu']:.0f}$", (i["k"], i["violacao"]), xytext=(6, 4),
                     textcoords="offset points", fontsize=7)
    ax1.axhline(1e-10, color="0.5", lw=0.8, ls=":")
    ax1.set_xlabel("iteração externa $k$")
    ax1.legend()
    ax2.semilogy(k, [abs(i["lam"] - ref["slsqp"]["lam"]) for i in h], "o-", color=COR["al"],
                 label=r"$|\hat\lambda_{k+1} - \hat\lambda_{SLSQP}|$")
    ax2.semilogy(k, [abs(i["lam"] - ref["tiro"]["lam"]) for i in h], "x--", color=COR["tiro"],
                 label=r"$|\hat\lambda_{k+1} - \hat\lambda_{EDO}|$")
    ax2.set_xlabel("iteração externa $k$")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.set_xticks(k)
    salvar(fig, "historico_al.pdf")


def figura_lambda_V(dados):
    lv = dados["lambda_V"]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot([i["V"] for i in lv], [i["lam_tiro"] for i in lv], "-", color=COR["tiro"],
            label=r"EDO: $\hat\lambda = (A - B\,\hat d)/\pi$")
    ax.plot([i["V"] for i in lv], [i["lam_al"] for i in lv], "x", color=COR["al"], ms=7,
            label=r"Lagrangiano aumentado ($N = 64$)")
    ax.set_xlabel(r"$\hat V$")
    ax.set_ylabel(r"$\hat\lambda$")
    ax.legend()
    salvar(fig, "lambda_volume.pdf")


def figura_descolamento(dados):
    desc = dados["descolamento"]
    tiro = next(c for c in dados["comparacao"] if c["nome"] == "V*")["tiro"]
    fig, ax = plt.subplots(figsize=(5.0, 6.4))
    x, z = perfil(tiro["r"], tiro["z"])
    ax.plot(x, z, "-", color=COR["tiro"], lw=1.2, label="EDO (tiro)")
    for nome, cor, rot in (("mu1000", COR["slsqp"], r"$\mu_0 = 1000$"), ("mu10", COR["al"], r"$\mu_0 = 10$")):
        g = desc[nome]
        x, z = perfil(g["r"], g["z"])
        ax.plot(x, z, "o-", color=cor, ms=3, lw=1, mfc="none",
                label=rot + rf", $\hat E = {g['energia']:.3f}$")
    bico(ax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\hat r$")
    ax.set_ylabel(r"$\hat z$")
    ax.legend(loc="lower right", fontsize=7)
    salvar(fig, "descolamento.pdf")


def eixos_iguais(ax, pts, zoom=0.85):
    meio = (pts.max(axis=0) + pts.min(axis=0)) / 2
    raio = np.ptp(pts, axis=0).max() / 2
    for eixo, c in zip("xyz", meio):
        getattr(ax, f"set_{eixo}lim")(c - raio, c + raio)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)


def figura_3d():
    fig = plt.figure(figsize=(10.0, 4.2))
    casos = (("circular", "bico circular"), ("quadrado", "bico quadrado"),
             ("parede", rf"bico circular e parede $\hat x \leq {PAREDE:g}$"))
    for i, (nome, titulo) in enumerate(casos):
        m = np.load(DADOS / f"gota3d_{nome}.npz")
        pts, tri = m["points"], m["connectivity"]
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(pts[tri], facecolor="lightsteelblue",
                                             edgecolor="0.25", linewidth=0.25, alpha=0.9))
        if nome == "parede":
            yy, zz = np.meshgrid([-2.0, 2.0], [pts[:, 2].min(), 0.0])
            ax.plot_surface(np.full_like(yy, PAREDE), yy, zz, color="tab:orange", alpha=0.25)
        eixos_iguais(ax, pts)
        ax.view_init(elev=15, azim=-60)
        ax.set_xlabel(r"$\hat x$")
        ax.set_ylabel(r"$\hat y$")
        ax.set_zlabel(r"$\hat z$")
        ax.set_title(titulo, fontsize=9)
    salvar(fig, "gota3d.pdf")


def figura_parede(dados):
    """Vistas de baixo e de frente da gota com e sem a parede."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.4))
    for nome, cor, rot in (("circular", COR["slsqp"], "sem parede"), ("parede", COR["al"], "com parede")):
        P = np.load(DADOS / f"gota3d_{nome}.npz")["points"]
        ax1.plot(P[:, 0], P[:, 1], ".", color=cor, ms=3, label=rot)
        sel = np.abs(P[:, 1]) < 1e-9          # vértices no plano y = 0
        for lado, r in ((1, rot), (-1, None)):  # cada metade, do bico ao ápice
            metade = sel & (lado * P[:, 0] >= -1e-12)
            o = np.argsort(-P[metade, 2])
            ax2.plot(P[metade, 0][o], P[metade, 2][o], "o-", color=cor, ms=2.5, lw=1, mfc="none", label=r)
    for ax in (ax1, ax2):
        ax.axvline(PAREDE, color="tab:orange", lw=2, label="parede")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\hat x$")
    ax1.set_ylabel(r"$\hat y$")
    ax1.set_title("vista de baixo", fontsize=9)
    ax2.set_ylabel(r"$\hat z$")
    ax2.set_title(r"corte no plano $\hat y = 0$", fontsize=9)
    ax2.legend(loc="lower left", fontsize=7)
    salvar(fig, "parede.pdf")


def figura_eixo3d(dados):
    """Vértices da gota 3D (bico circular) no plano (r, z) sobre o perfil da EDO,
    e erro em relação à EDO conforme a malha é refinada."""
    eixo = dados["eixo3d"]
    fig, axs = plt.subplots(1, 3, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    for ax, caso in zip(axs[:2], eixo):
        t = caso["tiro"]
        x, z = perfil(t["r"], t["z"])
        ax.plot(x, z, "-", color=COR["tiro"], lw=1.2, label=ROTULO["tiro"])
        g = caso["malhas"][-1]
        ax.plot(g["r_vertices"], g["z_vertices"], "o", color=COR["slsqp"], ms=2.5, mfc="none",
                label="3D, malha {}$\\times${}".format(*g["malha"]))
        bico(ax)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\hat r$")
        ax.set_title(rf"$\hat V = {caso['V']:g}$", fontsize=9)
    axs[0].set_ylabel(r"$\hat z$")
    axs[0].legend(loc="lower left", fontsize=7)
    ax = axs[2]
    for caso, marca in zip(eixo, ("o", "s")):
        n = np.array([g["n_variaveis"] for g in caso["malhas"]], dtype=float)
        ed = [abs(g["altura"] - caso["tiro"]["d"]) for g in caso["malhas"]]
        ee = [abs(g["energia"] - caso["tiro"]["energia"]) for g in caso["malhas"]]
        ax.loglog(n, ed, marca + "-", color=COR["slsqp"], mfc="none",
                  label=rf"$|\hat d - \hat d_{{EDO}}|$, $\hat V = {caso['V']:g}$")
        ax.loglog(n, ee, marca + "--", color=COR["al"], mfc="none",
                  label=rf"$|\hat E - \hat E_{{EDO}}|$, $\hat V = {caso['V']:g}$")
    n = np.array([g["n_variaveis"] for g in eixo[0]["malhas"]], dtype=float)
    e0 = abs(eixo[0]["malhas"][0]["altura"] - eixo[0]["tiro"]["d"])
    ax.loglog(n, e0 * (n / n[0])**-1, ":", color="0.4", label=r"inclinação $n^{-1}$")
    ax.set_xlabel("número de variáveis $n$")
    ax.set_ylabel("diferença para a EDO")
    ax.legend(fontsize=6.5)
    fig.subplots_adjust(wspace=0.35)
    salvar(fig, "eixo3d.pdf")


def figura_slsqp_livro(dados):
    ex = dados["exemplo_livro_slsqp"]
    it = np.asarray(ex["iteracoes"])
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(2 * np.cos(t), np.sin(t), "k-", lw=1.2, label=r"$h(x) = x_1^2/4 + x_2^2 - 1 = 0$")
    X, Y = np.meshgrid(np.linspace(-7, 3, 200), np.linspace(-5, 2.5, 200))
    cs = ax.contour(X, Y, X + 2 * Y, levels=12, colors="0.7", linewidths=0.7)
    ax.clabel(cs, fontsize=6)
    ax.plot(it[:, 0], it[:, 1], "o-", color=COR["slsqp"], ms=3.5, lw=1, label="iterados do SLSQP")
    ax.plot(*it[0], "s", color=COR["slsqp"], ms=6)
    ax.plot(-np.sqrt(2), -np.sqrt(2) / 2, "*", color=COR["al"], ms=12, label=r"$x^*$")
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(fontsize=7, loc="upper left")
    salvar(fig, "slsqp_livro.pdf")


# ---------------------------------------------------------------------------
# tabelas
# ---------------------------------------------------------------------------

def num(x, fmt):
    return rf"\num{{{format(x, fmt)}}}"


def tabela_comparacao(dados):
    nomes = {"V*/3": r"$\hat V^*/3$", "2V*/3": r"$2\hat V^*/3$", "V*": r"$\hat V^*$"}
    linhas = []
    for caso in dados["comparacao"]:
        t, s, a = caso["tiro"], caso["slsqp"], caso["al"]
        linhas += [
            rf"{nomes[caso['nome']]} & EDO (tiro) & {num(t['d'], '.6f')} & {num(t['lam'], '.6f')} & --- & --- \\",
            rf" & SLSQP & {num(s['d'], '.6f')} & {num(s['lam'], '.6f')} & {s['iteracoes']} & {num(s['tempo'], '.2f')} \\",
            rf" & Lagr. aumentado & {num(a['d'], '.6f')} & {num(a['lam'], '.6f')} & {a['externas']} ({a['internas']}) & {num(a['tempo'], '.2f')} \\",
        ]
        linhas.append(r"\midrule")
    linhas[-1] = r"\bottomrule"
    corpo = "\n".join(linhas)
    (TABELAS / "comparacao.tex").write_text(
        "\\begin{tabular}{llcccc}\n\\toprule\n"
        "$\\hat V$ & Método & $\\hat d$ & $\\hat\\lambda$ & Iterações & Tempo (s) \\\\\n\\midrule\n"
        f"{corpo}\n\\end{{tabular}}\n", encoding="utf-8")


def tabela_convergencia(dados):
    conv = dados["convergencia"]
    ref = next(c for c in dados["comparacao"] if c["nome"] == "2V*/3")["tiro"]
    linhas = []
    for N, s, a in zip(conv["N"], conv["slsqp"], conv["al"]):
        linhas.append(
            rf"{N} & {num(abs(s['d'] - ref['d']), '.2e')} & {num(abs(a['d'] - ref['d']), '.2e')} & "
            rf"{num(abs(s['lam'] - ref['lam']), '.2e')} & {num(abs(a['lam'] - ref['lam']), '.2e')} & "
            rf"{num(s['tempo'], '.2f')} & {num(a['tempo'], '.2f')} \\")
    (TABELAS / "convergencia.tex").write_text(
        "\\begin{tabular}{rcccccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{$|\\hat d - \\hat d_{EDO}|$} & \\multicolumn{2}{c}{$|\\hat\\lambda - \\hat\\lambda_{EDO}|$}"
        " & \\multicolumn{2}{c}{Tempo (s)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "$N$ & SLSQP & L.A. & SLSQP & L.A. & SLSQP & L.A. \\\\\n\\midrule\n"
        + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def tabela_exemplo_livro(dados):
    livro = {0: ("0.50", "0.000"), 2: ("0.61", "1.146"), 9: ("1.18", "1.413")}
    h = dados["exemplo_livro"]["historico"]
    linhas = [rf"{k} & {num(float(m), '.2f')} & {num(float(l), '.3f')} & "
              rf"{num(h[k]['mu'], '.4f')} & {num(h[k]['lam'], '.6f')} \\"
              for k, (m, l) in livro.items()]
    ex = dados["exemplo_livro"]
    linhas.append(r"\midrule")
    linhas.append(rf"{ex['nit']} (final) & --- & $\sqrt 2 \approx \num{{1.414214}}$ & --- & "
                  rf"{num(ex['lam'], '.6f')} \\")
    (TABELAS / "exemplo_livro.tex").write_text(
        "\\begin{tabular}{rcccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Livro (Fig. 5.32)} & \\multicolumn{2}{c}{Este trabalho} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "$k$ & $\\mu_k$ & $\\lambda_k$ & $\\mu_k$ & $\\lambda_k$ \\\\\n\\midrule\n"
        + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def tabela_3d(dados):
    linhas = []
    for nome, rot in (("circular", "circular"), ("quadrado", "quadrado"), ("parede", "circular + parede")):
        g = dados["gota3d"][nome]
        linhas.append(rf"{rot} & {g['n_variaveis']} & {g['iteracoes']} & {num(g['tempo'], '.2f')} & "
                      rf"{num(g['altura'], '.4f')} & {num(g['energia'], '.4f')} & {num(g['lam'], '.4f')} \\")
    (TABELAS / "gota3d.tex").write_text(
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "Bico & Variáveis & Iterações & Tempo (s) & Altura $\\hat d$ & Energia $\\hat E$ & $\\hat\\lambda$ \\\\\n"
        "\\midrule\n" + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def tabela_eixo3d(dados):
    linhas = []
    for caso in dados["eixo3d"]:
        t = caso["tiro"]
        linhas.append(rf"\multirow{{{len(caso['malhas']) + 1}}}{{*}}{{${caso['V']:g}$}} & EDO (tiro) & --- & --- & "
                      rf"{num(t['d'], '.4f')} & {num(t['energia'], '.4f')} & {num(t['lam'], '.4f')} \\")
        for g in caso["malhas"]:
            linhas.append(rf" & {g['malha'][0]}$\times${g['malha'][1]} & {g['n_variaveis']} & {num(g['tempo'], '.2f')} & "
                          rf"{num(g['altura'], '.4f')} & {num(g['energia'], '.4f')} & {num(g['lam'], '.4f')} \\")
        linhas.append(r"\midrule")
    linhas[-1] = r"\bottomrule"
    (TABELAS / "eixo3d.tex").write_text(
        "\\begin{tabular}{llccccc}\n\\toprule\n"
        "$\\hat V$ & Malha & Variáveis & Tempo (s) & $\\hat d$ & $\\hat E$ & $\\hat\\lambda$ \\\\\n\\midrule\n"
        + "\n".join(linhas) + "\n\\end{tabular}\n", encoding="utf-8")


def tabela_3d_convergencia(dados):
    """Erro relativo da gota 3D (bico circular) em relação à EDO e ordem observada.
    Com n variáveis, o tamanho dos triângulos é h ~ n^(-1/2), então a ordem em h é
    p = 2 log(e_k / e_{k+1}) / log(n_{k+1} / n_k)."""
    def ordem(e0, e1, n0, n1):
        return 2.0 * np.log(e0 / e1) / np.log(n1 / n0)
    linhas = []
    for caso in dados["eixo3d"]:
        t, ms = caso["tiro"], caso["malhas"]
        ed = [abs(g["altura"] - t["d"]) / t["d"] for g in ms]
        el = [abs(g["lam"] - t["lam"]) / t["lam"] for g in ms]
        for k, g in enumerate(ms):
            pd = "---" if k == 0 else num(ordem(ed[k - 1], ed[k], ms[k - 1]["n_variaveis"], g["n_variaveis"]), ".2f")
            pl = "---" if k == 0 else num(ordem(el[k - 1], el[k], ms[k - 1]["n_variaveis"], g["n_variaveis"]), ".2f")
            v = rf"\multirow{{{len(ms)}}}{{*}}{{${caso['V']:g}$}}" if k == 0 else ""
            linhas.append(rf"{v} & {g['malha'][0]}$\times${g['malha'][1]} & {g['n_variaveis']} & "
                          rf"{num(100 * ed[k], '.2f')} & {pd} & {num(100 * el[k], '.2f')} & {pl} \\")
        linhas.append(r"\midrule")
    linhas[-1] = r"\bottomrule"
    (TABELAS / "eixo3d_erros.tex").write_text(
        "\\begin{tabular}{llccccc}\n\\toprule\n"
        " & & & \\multicolumn{2}{c}{Altura $\\hat d$} & \\multicolumn{2}{c}{Pressão $\\hat\\lambda$} \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "$\\hat V$ & Malha & $n$ & erro (\\%) & ordem & erro (\\%) & ordem \\\\\n\\midrule\n"
        + "\n".join(linhas) + "\n\\end{tabular}\n", encoding="utf-8")


def tabela_parede(dados):
    livre, parede = dados["gota3d"]["circular"], dados["gota3d"]["parede"]
    linhas = [
        rf"sem parede & {num(livre['altura'], '.4f')} & {num(livre['x_max'], '.3f')} & {num(livre['energia'], '.4f')} & "
        rf"{num(livre['lam'], '.4f')} & --- & --- & --- \\",
        rf"com parede & {num(parede['altura'], '.4f')} & {num(parede['x_max'], '.3f')} & {num(parede['energia'], '.4f')} & "
        rf"{num(parede['lam'], '.4f')} & {parede['ativas']} de {parede['n_restricoes_parede']} & "
        rf"{num(parede['sigma_max'], '.4f')} & {num(parede['sigma_soma'], '.4f')} \\",
    ]
    (TABELAS / "parede.tex").write_text(
        "\\begin{tabular}{lccccccc}\n\\toprule\n"
        "Caso & $\\hat d$ & $\\hat x_{\\max}$ & $\\hat E$ & $\\hat\\lambda$ & Ativas & "
        "$\\max_j \\sigma_j$ & $\\sum_j \\sigma_j$ \\\\\n\\midrule\n"
        + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recalcular", action="store_true", help="refaz todas as contas")
    args = parser.parse_args()
    for pasta in (FIGURAS, TABELAS, DADOS):
        pasta.mkdir(exist_ok=True)

    arquivo = DADOS / "resultados.json"
    if args.recalcular or not arquivo.exists():
        t = time.perf_counter()
        dados = calcular()
        dados["tempo_total"] = time.perf_counter() - t
        arquivo.write_text(json.dumps(dados, indent=1, ensure_ascii=False), encoding="utf-8")
    else:
        dados = json.loads(arquivo.read_text(encoding="utf-8"))

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                         "legend.fontsize": 8, "savefig.bbox": "tight"})
    figura_familia(dados)
    figura_curva_VA(dados)
    figura_comparacao(dados)
    figura_convergencia(dados)
    figura_historico(dados)
    figura_lambda_V(dados)
    figura_descolamento(dados)
    figura_3d()
    figura_parede(dados)
    figura_eixo3d(dados)
    figura_slsqp_livro(dados)
    tabela_comparacao(dados)
    tabela_convergencia(dados)
    tabela_exemplo_livro(dados)
    tabela_3d(dados)
    tabela_eixo3d(dados)
    tabela_3d_convergencia(dados)
    tabela_parede(dados)
    print("figuras em", FIGURAS, "e tabelas em", TABELAS)


if __name__ == "__main__":
    main()
