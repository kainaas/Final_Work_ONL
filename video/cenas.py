"""
Cenas do vídeo (Manim Community, v0.18 ou mais nova).

Antes, gere os dados:          python gerar_dados.py
Prévia rápida de uma cena:     manim -pql cenas.py GotaSeAjeitando
Versão final (1080p, 60 fps):  manim -qh cenas.py GotaSeAjeitando
Todas as cenas:                manim -qh -a cenas.py

As cenas usam só Text (Pango), então não precisam de LaTeX. A ordem e o texto
narrado de cada cena estão em roteiro.md.
"""
from pathlib import Path

import numpy as np
from manim import *  # noqa: F401,F403

DADOS = Path(__file__).resolve().parent / "dados"

AGUA = "#4FA3E0"
AGUA_ESCURA = "#1F5F99"
BICO = GREY_B
DESTAQUE = "#F2A33A"
FUNDO = "#10141A"

config.background_color = FUNDO


def carregar(nome):
    return np.load(DADOS / f"{nome}.npz")


# ---------------------------------------------------------------------------
# utilidades 2D: perfil (2, N) do bico (z = 0) até o ápice -> forma na tela
# ---------------------------------------------------------------------------

def contorno(P, escala=1.0, origem=ORIGIN):
    """Pontos na tela da gota inteira (as duas metades do perfil)."""
    r, z = np.asarray(P[0]), np.asarray(P[1])
    x = np.concatenate([-r[::-1], r[1:]])
    y = np.concatenate([z[::-1], z[1:]])
    return [origem + escala * np.array([xi, yi, 0.0]) for xi, yi in zip(x, y)]


def gota_2d(P, escala=1.0, origem=ORIGIN, suave=True, cor=AGUA):
    pts = contorno(P, escala, origem)
    forma = VMobject(stroke_color=cor, stroke_width=4, fill_color=cor, fill_opacity=0.35)
    if suave:
        forma.set_points_smoothly(pts)
    else:
        forma.set_points_as_corners(pts)
    return forma


def bico_2d(escala=1.0, origem=ORIGIN, altura=0.6):
    """Duas paredes do bico, de raio 1, acima do plano z = 0."""
    lados = VGroup()
    for x0 in (1.0, -1.25):
        r = Rectangle(width=0.25 * escala, height=altura * escala, stroke_width=0,
                      fill_color=BICO, fill_opacity=1.0)
        r.move_to(origem + escala * np.array([x0 + 0.125, altura / 2, 0.0]))
        lados.add(r)
    return lados


def interpola(perfis, t):
    """Perfil intermediário entre os quadros inteiros vizinhos de t."""
    i = int(np.clip(np.floor(t), 0, len(perfis) - 2))
    a = float(np.clip(t - i, 0.0, 1.0))
    return (1 - a) * perfis[i] + a * perfis[i + 1]


def titulo(texto):
    return Text(texto, font_size=40).to_edge(UP)


# ---------------------------------------------------------------------------
# 1. Abertura
# ---------------------------------------------------------------------------

class Abertura(Scene):
    def construct(self):
        d = carregar("crescendo")
        perfis = d["perfis"][:-1]
        esc, origem = 0.75, UP * 2.2
        t = ValueTracker(0.0)
        gota = always_redraw(lambda: gota_2d(interpola(perfis, t.get_value()), esc, origem))
        nome = Text("Qual é a forma\nde uma gota?", font_size=52, line_spacing=1.2).to_edge(LEFT, buff=1.0)
        self.add(bico_2d(esc, origem), gota)
        self.play(Write(nome), run_time=1.5)
        self.play(t.animate.set_value(len(perfis) - 1), run_time=8, rate_func=smooth)
        self.wait()


# ---------------------------------------------------------------------------
# 2. A pele da água: mesma área, bordas diferentes
# ---------------------------------------------------------------------------

class PeleDaAgua(Scene):
    def construct(self):
        self.add(titulo("Mesma área, bordas diferentes"))
        area = np.pi                      # as três figuras têm área pi
        circulo = Circle(radius=1.0)
        elipse = Ellipse(width=2 * 2.0, height=2 * 0.5)           # a b = 1
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        r = 1.0 + 0.3 * np.cos(3 * t) + 0.15 * np.sin(5 * t)
        k = np.sqrt(area / (0.5 * np.sum(r**2) * (2 * np.pi / len(t))))
        irregular = Polygon(*[k * np.array([ri * np.cos(ti), ri * np.sin(ti), 0]) for ri, ti in zip(r, t)])
        formas = VGroup(irregular, elipse, circulo)
        for f in formas:
            f.set_stroke(AGUA, 5).set_fill(AGUA, 0.3)
        formas.arrange(RIGHT, buff=1.2).shift(DOWN * 0.3)

        def perimetro(m):
            p = m.get_anchors()
            return float(np.sum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)))

        rotulos = VGroup(*[Text(f"borda: {perimetro(f):.2f}", font_size=28).next_to(f, DOWN, buff=0.5)
                           for f in formas])
        self.play(LaggedStart(*[DrawBorderThenFill(f) for f in formas], lag_ratio=0.3))
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.2) for r in rotulos], lag_ratio=0.3))
        self.play(Indicate(VGroup(circulo, rotulos[2]), color=DESTAQUE, scale_factor=1.1))
        self.wait(2)


# ---------------------------------------------------------------------------
# 3. Cabo de guerra: energia da pele x energia da altura
# ---------------------------------------------------------------------------

class CaboDeGuerra(Scene):
    def construct(self):
        d = carregar("cabo_de_guerra")
        formas, partes, fatores = d["formas"], d["partes"], d["fatores"]
        sup, grav, tot = partes[:, 0], partes[:, 1], partes[:, 2]
        esc, origem = 0.6, LEFT * 3.5 + UP * 2.3
        t = ValueTracker(float(np.argmin(np.abs(fatores - 1.0))))
        self.add(bico_2d(esc, origem))
        gota = always_redraw(lambda: gota_2d(interpola(formas, t.get_value()), esc, origem))
        self.add(gota)

        eixos = Axes(x_range=[fatores[0], fatores[-1], 0.2], y_range=[float(tot.min()) - 1, float(tot.max()) + 1, 2],
                     x_length=6, y_length=4, tips=False).to_edge(RIGHT, buff=0.6).shift(DOWN * 0.4)
        rot_x = Text("quanto a gota está esticada", font_size=24).next_to(eixos, DOWN, buff=0.2)
        rot_y = Text("energia total", font_size=24).next_to(eixos, UP, buff=0.2)
        curva = eixos.plot_line_graph(fatores, tot, add_vertex_dots=False, line_color=DESTAQUE, stroke_width=4)
        ponto = always_redraw(lambda: Dot(eixos.c2p(*self._ponto(fatores, tot, t.get_value())), color=WHITE))

        # barras: pele (sobe) e altura (desce)
        base = LEFT * 0.6 + DOWN * 2.8
        def barras():
            i = t.get_value()
            s = np.interp(i, np.arange(len(sup)), sup) - sup.min() + 0.3
            g = -(np.interp(i, np.arange(len(grav)), grav) - grav.max()) + 0.3
            b1 = Rectangle(width=0.5, height=0.25 * s, fill_color=AGUA, fill_opacity=0.9, stroke_width=0)
            b2 = Rectangle(width=0.5, height=0.25 * g, fill_color=DESTAQUE, fill_opacity=0.9, stroke_width=0)
            b1.move_to(base, aligned_edge=DOWN)
            b2.move_to(base + RIGHT * 0.8, aligned_edge=DOWN)
            return VGroup(b1, b2)
        legenda = VGroup(Text("pele", font_size=22, color=AGUA), Text("gravidade", font_size=22, color=DESTAQUE))
        legenda[0].next_to(base, DOWN, buff=0.15)
        legenda[1].next_to(base + RIGHT * 0.8, DOWN, buff=0.15)

        self.play(Create(eixos), FadeIn(rot_x), FadeIn(rot_y))
        self.add(always_redraw(barras), legenda)
        self.play(Create(curva), FadeIn(ponto))
        self.play(t.animate.set_value(0), run_time=3)
        self.play(t.animate.set_value(len(fatores) - 1), run_time=5)
        self.play(t.animate.set_value(float(np.argmin(tot))), run_time=3)
        seta = Arrow(eixos.c2p(fatores[np.argmin(tot)], tot.min() + 1.6), eixos.c2p(fatores[np.argmin(tot)], tot.min() + 0.2),
                     color=WHITE, buff=0)
        self.play(GrowArrow(seta), Write(Text("equilíbrio", font_size=26).next_to(seta, UP, buff=0.1)))
        self.wait(2)

    @staticmethod
    def _ponto(fatores, tot, i):
        k = np.arange(len(fatores))
        return float(np.interp(i, k, fatores)), float(np.interp(i, k, tot))


# ---------------------------------------------------------------------------
# 4. Otimização: descer o morro e andar na trilha
# ---------------------------------------------------------------------------

class DescendoOMorro(Scene):
    def construct(self):
        eixos = Axes(x_range=[-3, 3, 1], y_range=[-1, 5, 1], x_length=10, y_length=5, tips=False,
                     axis_config={"include_ticks": False}).shift(DOWN * 0.5)
        f = lambda x: 0.25 * x**4 - 0.6 * x**2 + 0.35 * x + 1.5    # noqa: E731
        df = lambda x: x**3 - 1.2 * x + 0.35                         # noqa: E731
        morro = eixos.plot(f, x_range=[-2.4, 2.4], color=GREEN_C, stroke_width=5)
        neblina = Rectangle(width=14, height=8, fill_color=FUNDO, fill_opacity=0.55, stroke_width=0)
        self.play(Create(eixos), Create(morro))
        self.add(neblina)
        x = 2.2
        bola = Dot(eixos.c2p(x, f(x)), radius=0.14, color=DESTAQUE)
        self.add(bola)
        for _ in range(12):
            novo = x - 0.18 * df(x)
            self.play(bola.animate.move_to(eixos.c2p(novo, f(novo))), run_time=0.45)
            x = novo
        self.play(FadeOut(neblina))
        regra = Text("regra: a quantidade de água não muda", font_size=32, color=DESTAQUE).to_edge(UP)
        self.play(Write(regra))
        self.wait(2)


class TrilhaRestrita(Scene):
    def construct(self):
        it = carregar("trilha")["iteracoes"]
        plano = NumberPlane(x_range=[-7, 3, 1], y_range=[-5, 2.5, 1], x_length=10, y_length=7.5,
                            background_line_style={"stroke_opacity": 0.15})
        # curvas de nível de x1 + 2 x2 (retas) e a trilha (elipse)
        niveis = VGroup(*[plano.plot(lambda x, c=c: (c - x) / 2, x_range=[-7, 3], color=GREY_C,
                                     stroke_width=1.5, stroke_opacity=0.6) for c in range(-12, 6, 2)])
        trilha = plano.plot_parametric_curve(lambda s: np.array([2 * np.cos(s), np.sin(s)]), t_range=[0, TAU],
                                             color=GREEN_C, stroke_width=6)
        rot = Text("a trilha (restrição)", font_size=26, color=GREEN_C).next_to(plano.c2p(2, 1), UR, buff=0.1)
        alvo = Star(color=DESTAQUE, fill_opacity=1).scale(0.18).move_to(plano.c2p(-np.sqrt(2), -np.sqrt(2) / 2))
        desce = Text("mais baixo  →", font_size=24, color=GREY_B).rotate(-np.arctan(2) - PI / 2).move_to(plano.c2p(-5.5, 1.5))
        self.play(FadeIn(plano), Create(niveis), FadeIn(desce))
        self.play(Create(trilha), FadeIn(rot))
        bola = Dot(plano.c2p(*it[0]), radius=0.12, color=AGUA)
        self.add(bola)
        for p0, p1 in zip(it[:-1], it[1:]):
            seg = Line(plano.c2p(*p0), plano.c2p(*p1), stroke_color=AGUA, stroke_width=3)
            self.play(Create(seg), bola.animate.move_to(plano.c2p(*p1)), run_time=0.7)
        self.play(FadeIn(alvo, scale=2))
        self.wait(2)


# ---------------------------------------------------------------------------
# 5. O computador procura a gota
# ---------------------------------------------------------------------------

class GotaSeAjeitando(Scene):
    def construct(self):
        d = carregar("otimizacao_2d")
        perfis, energias, volumes, edo = d["perfis"], d["energias"], d["volumes"], d["edo"]
        esc, origem = 0.75, LEFT * 2.5 + UP * 2.6
        t = ValueTracker(0.0)

        def pontos():
            P = interpola(perfis, t.get_value())
            g = VGroup(gota_2d(P, esc, origem, suave=False))
            g.add(*[Dot(origem + esc * np.array([r, z, 0]), radius=0.05, color=WHITE) for r, z in P.T])
            return g

        self.add(bico_2d(esc, origem))
        gota = always_redraw(pontos)
        self.play(FadeIn(gota))

        k = np.arange(len(perfis))
        rot_e = Text("energia", font_size=30).move_to(RIGHT * 3 + UP * 1.5)
        rot_v = Text("volume", font_size=30).move_to(RIGHT * 3 + DOWN * 0.2)
        num_e = always_redraw(lambda: DecimalNumber(np.interp(t.get_value(), k, energias[:, 2]), num_decimal_places=3,
                                                    font_size=44, color=DESTAQUE).next_to(rot_e, DOWN))
        num_v = always_redraw(lambda: DecimalNumber(np.interp(t.get_value(), k, volumes), num_decimal_places=3,
                                                    font_size=44, color=AGUA).next_to(rot_v, DOWN))
        passo = always_redraw(lambda: Text(f"passo {int(round(t.get_value())) + 1}", font_size=28)
                              .move_to(RIGHT * 3 + DOWN * 2.2))
        self.play(FadeIn(rot_e), FadeIn(rot_v), FadeIn(num_e), FadeIn(num_v), FadeIn(passo))
        self.play(t.animate.set_value(len(perfis) - 1), run_time=10, rate_func=linear)
        self.wait()
        curva = gota_2d(edo, esc, origem, cor=DESTAQUE).set_fill(opacity=0).set_stroke(width=3)
        rot = Text("equação da física", font_size=26, color=DESTAQUE).next_to(curva, DOWN)
        self.play(Create(curva), FadeIn(rot))
        self.wait(2)


# ---------------------------------------------------------------------------
# 6. Crescendo até cair
# ---------------------------------------------------------------------------

class CrescendoAteCair(Scene):
    def construct(self):
        d = carregar("crescendo")
        perfis, volumes, solta = d["perfis"], d["volumes"], d["solta"]
        esc, origem = 0.62, LEFT * 3 + UP * 2.8
        t = ValueTracker(0.0)
        k = np.arange(len(perfis))
        self.add(bico_2d(esc, origem))
        gota = always_redraw(lambda: gota_2d(interpola(perfis, t.get_value()), esc, origem))
        rot = Text("volume", font_size=30).move_to(RIGHT * 2 + UP * 2)
        num = always_redraw(lambda: DecimalNumber(np.interp(t.get_value(), k, volumes), num_decimal_places=1,
                                                  font_size=48, color=AGUA).next_to(rot, DOWN))
        self.add(gota, rot, num)
        self.play(t.animate.set_value(len(perfis) - 1), run_time=9, rate_func=linear)
        limite = Text("volume máximo", font_size=30, color=DESTAQUE).next_to(num, DOWN, buff=0.4)
        self.play(FadeIn(limite), Flash(gota.get_bottom(), color=DESTAQUE))
        self.wait()
        origem2 = RIGHT * 3.6 + UP * 2.8
        outra = gota_2d(solta, esc, origem2, suave=False, cor=DESTAQUE)
        self.play(FadeIn(bico_2d(esc, origem2)), Create(outra), run_time=2)
        self.play(FadeIn(Text("menos energia: a gota se solta", font_size=26, color=DESTAQUE)
                         .next_to(outra, DOWN, buff=0.3)))
        self.wait(2)


# ---------------------------------------------------------------------------
# 7. Em três dimensões
# ---------------------------------------------------------------------------

ESC_3D = 0.9


def malha_3d(points, connectivity, cor=AGUA, opacidade=0.85):
    """VGroup de triângulos; o eixo z da gota é o eixo vertical da tela."""
    P = ESC_3D * np.asarray(points) + np.array([0, 0, 1.6])
    grupo = VGroup()
    for a, b, c in connectivity:
        tri = Polygon(P[a], P[b], P[c], stroke_color=AGUA_ESCURA, stroke_width=0.6,
                      fill_color=cor, fill_opacity=opacidade)
        grupo.add(tri)
    return grupo


def mover_malha(grupo, points, connectivity):
    P = ESC_3D * np.asarray(points) + np.array([0, 0, 1.6])
    for tri, (a, b, c) in zip(grupo, connectivity):
        tri.set_points_as_corners([P[a], P[b], P[c], P[a]])


class Gota3D(ThreeDScene):
    def construct(self):
        d = carregar("gota3d_iteracoes")
        quadros, tri = d["points"], d["connectivity"]
        # no máximo ~80 quadros, sempre com o último
        idx = np.unique(np.concatenate([np.linspace(0, len(quadros) - 1, 80).astype(int), [len(quadros) - 1]]))
        quadros = quadros[idx]
        self.set_camera_orientation(phi=72 * DEGREES, theta=-60 * DEGREES, zoom=0.9)
        gota = malha_3d(quadros[0], tri)
        self.play(Create(gota), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(2)
        t = ValueTracker(0.0)

        def atualiza(g):
            mover_malha(g, interpola(quadros, t.get_value()), tri)
        gota.add_updater(atualiza)
        self.play(t.animate.set_value(len(quadros) - 1), run_time=10, rate_func=smooth)
        gota.remove_updater(atualiza)
        self.wait(4)
        self.stop_ambient_camera_rotation()


class Comparacao3D(Scene):
    def construct(self):
        d = carregar("gota3d_iteracoes")
        P, edo = d["points"][-1], d["edo"]
        esc, origem = 0.75, UP * 2.6
        self.add(titulo("3D (pontos) e equação (linha)").scale(0.8))
        self.add(bico_2d(esc, origem))
        linha = gota_2d(edo, esc, origem, cor=DESTAQUE).set_fill(opacity=0).set_stroke(width=4)
        r = np.hypot(P[:, 0], P[:, 1])
        lado = np.where(P[:, 0] >= 0, 1.0, -1.0)
        pts = VGroup(*[Dot(origem + esc * np.array([s * ri, zi, 0]), radius=0.035, color=AGUA)
                       for s, ri, zi in zip(lado, r, P[:, 2])])
        self.play(Create(linha))
        self.play(LaggedStart(*[FadeIn(p) for p in pts], lag_ratio=0.005, run_time=4))
        self.wait(2)


class QuadradoEParede(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=72 * DEGREES, theta=-60 * DEGREES, zoom=0.9)
        d = carregar("gota3d_quadrado")
        nome = Text("bico quadrado", font_size=36).to_corner(UL)
        self.add_fixed_in_frame_mobjects(nome)
        gota = malha_3d(d["points"], d["connectivity"])
        self.play(Create(gota), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(6)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(gota), FadeOut(nome))

        d = carregar("gota3d_parede")
        nome = Text("bico redondo encostado na parede", font_size=36).to_corner(UL)
        self.add_fixed_in_frame_mobjects(nome)
        gota = malha_3d(d["points"], d["connectivity"])
        zmin = ESC_3D * d["points"][:, 2].min() + 1.6
        parede = Polygon(*[np.array([ESC_3D * 1.0, y, z]) for y, z in
                           ((-2.2, zmin - 0.3), (2.2, zmin - 0.3), (2.2, 2.0), (-2.2, 2.0))],
                         stroke_width=0, fill_color=DESTAQUE, fill_opacity=0.25)
        self.set_camera_orientation(phi=72 * DEGREES, theta=-100 * DEGREES, zoom=0.9)
        self.play(FadeIn(parede), Create(gota), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(8)
        self.stop_ambient_camera_rotation()


# ---------------------------------------------------------------------------
# 8. Fechamento
# ---------------------------------------------------------------------------

class Fechamento(Scene):
    def construct(self):
        itens = VGroup(
            Text("• medir a tensão superficial de líquidos", font_size=32),
            Text("• impressoras a jato de tinta", font_size=32),
            Text("• conta-gotas de remédio", font_size=32),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).to_edge(LEFT, buff=1.0)
        self.add(titulo("Para que serve?"))
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.3) for i in itens], lag_ratio=0.6))
        self.wait(3)
        self.play(FadeOut(itens))
        d = carregar("otimizacao_2d")
        esc, origem = 0.75, UP * 2.2 + RIGHT * 3
        self.play(FadeIn(bico_2d(esc, origem)), DrawBorderThenFill(gota_2d(d["edo"], esc, origem)))
        fim = Text("A natureza também otimiza.", font_size=44, color=DESTAQUE).to_edge(LEFT, buff=0.8)
        self.play(Write(fim))
        self.wait(3)
