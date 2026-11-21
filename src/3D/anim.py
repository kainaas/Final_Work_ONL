import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def run_simulations(V_initial, V_final, num_frames, Bond, num_stacks,
                    num_sectors, nozzle, fix_boundary, QuadRule):
    '''
        Roda uma simulação por quadro, com o volume variando linearmente
        de V_initial até V_final. Retorna (volumes, lista de malhas).
    '''
    # import aqui para que o modo --load não precise de scipy/otimização
    from optimization_3d import find_droplet_optimization_3d

    volumes = np.linspace(V_initial, V_final, num_frames)
    meshes = []

    for k, V in enumerate(volumes):
        print(f"[{k + 1}/{num_frames}] V = {V:.4f}")
        m = find_droplet_optimization_3d(
            V, Bond, num_stacks, num_sectors, nozzle,
            fix_boundary=fix_boundary,
            QuadRule=QuadRule,
            verbose=False,
        )
        meshes.append(m)

    return volumes, meshes


def save_frames(path, volumes, meshes, **params):
    '''
        Salva todos os quadros num único .npz:
          volumes      (F,)       volume de cada quadro
          points       (F, N, 3)  vértices de cada quadro
          connectivity (T, 3)     triângulos (iguais em todos os quadros)
          cut_boundary (n,)       índices do anel da calota esférica
          nozzle_boundary (n,)    índices do anel do bico (se existir)
          + parâmetros escalares/arrays passados em **params (Bond, nozzle, ...)
    '''
    points = np.stack([np.asarray(m.points, dtype=float) for m in meshes])
    data = dict(
        volumes=np.asarray(volumes, dtype=float),
        points=points,
        connectivity=np.asarray(meshes[0].connectivity, dtype=int),
        cut_boundary=np.asarray(meshes[0].cut_boundary, dtype=int),
    )
    if getattr(meshes[0], "nozzle_boundary", None) is not None:
        data["nozzle_boundary"] = np.asarray(meshes[0].nozzle_boundary, dtype=int)
    data.update({k: np.asarray(v) for k, v in params.items()})

    np.savez_compressed(path, **data)
    print(f"Quadros salvos em {path}  (points: {points.shape})")


def load_frames(path):
    '''
        Lê um .npz gerado por save_frames.
        Retorna (volumes, points (F, N, 3), connectivity (T, 3)).
    '''
    with np.load(path) as d:
        return d["volumes"], d["points"], d["connectivity"]


def animate(volumes, all_points, connectivity, interval=200, save=None):
    '''
        Anima a sequência de quadros. A conectividade é a mesma em todos os
        quadros, então só os vértices são atualizados.
    '''
    all_points = np.asarray(all_points)
    connectivity = np.asarray(connectivity, dtype=int)
    num_frames = len(all_points)

    # limites fixos para todos os quadros
    stacked = all_points.reshape(-1, 3)
    center = (stacked.max(axis=0) + stacked.min(axis=0)) / 2
    half = np.ptp(stacked, axis=0).max() / 2

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    surface = Poly3DCollection(
        all_points[0][connectivity],
        facecolor="lightblue",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )
    ax.add_collection3d(surface)

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title = ax.set_title("")

    def update(k):
        surface.set_verts(all_points[k][connectivity])
        title.set_text(f"V = {volumes[k]:.4f}  (quadro {k + 1}/{num_frames})")
        return surface, title

    anim = FuncAnimation(
        fig, update, frames=num_frames, interval=interval, blit=False
    )

    if save:
        anim.save(save, dpi=120, fps=1000.0 / interval)
        print(f"Animação salva em {save}")

    plt.show()
    return anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animação da gota com volume crescente."
    )
    parser.add_argument("--v_initial", type=float, default=0.5,
                        help="volume inicial (adimensional)")
    parser.add_argument("--v_final", type=float, default=2.0,
                        help="volume final (adimensional)")
    parser.add_argument("--frames", type=int, default=20,
                        help="número de quadros (simulações)")
    parser.add_argument("--npz", type=str, default="droplet_frames.npz",
                        help="arquivo .npz onde salvar todos os quadros")
    parser.add_argument("--load", type=str, default=None,
                        help="carrega um .npz existente e só anima (sem simular)")
    parser.add_argument("--save", type=str, default=None,
                        help="arquivo de saída da animação (.gif ou .mp4); opcional")
    parser.add_argument("--interval", type=int, default=200,
                        help="tempo entre quadros em ms (menor = mais rápido)")
    parser.add_argument("--fps", type=float, default=None,
                        help="quadros por segundo; se dado, substitui --interval")
    args = parser.parse_args()

    if args.load:
        volumes, points, connectivity = load_frames(args.load)
    else:
        if args.frames < 1:
            raise ValueError("frames deve ser >= 1.")

        # mesmos parâmetros físicos do test.py
        gamma = 0.0728      # N/m
        density = 1000.0    # kg/m³
        a = np.sqrt(2) * 0.001
        g = 9.81
        Bond = a**2 * density * g / gamma

        num_stacks = 8
        num_sectors = 16
        QuadRule = "strang3"
        fix_boundary = False

        nozzle = np.array([
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, 1.0]
        ])

        volumes, meshes = run_simulations(
            args.v_initial, args.v_final, args.frames,
            Bond=Bond,
            num_stacks=num_stacks,
            num_sectors=num_sectors,
            nozzle=nozzle,
            fix_boundary=fix_boundary,
            QuadRule=QuadRule,
        )

        save_frames(
            args.npz, volumes, meshes,
            Bond=Bond, nozzle=nozzle,
            num_stacks=num_stacks, num_sectors=num_sectors,
            fix_boundary=fix_boundary, QuadRule=QuadRule,
        )

        points = np.stack([np.asarray(m.points) for m in meshes])
        connectivity = np.asarray(meshes[0].connectivity, dtype=int)

    interval = 1000.0 / args.fps if args.fps else args.interval

    animate(volumes, points, connectivity,
            interval=interval, save=args.save)