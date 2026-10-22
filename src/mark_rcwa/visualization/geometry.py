"""Structure view showing the actual staircase solved by RCWA."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


def plot_geometry(stack, path, materials=None, wavelength_um=.633):
    names = sorted({name for layer in stack.layers for name in layer.materials})
    palette = plt.get_cmap('tab10')
    colors = {name: palette(i % 10) for i, name in enumerate(names)}
    if materials is not None:
        for name in names:
            materials.index(name, wavelength_um)
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    z = 0.
    for layer in stack.layers:
        for left, right, name in zip(layer.edges_um[:-1], layer.edges_um[1:], layer.materials):
            ax.add_patch(Rectangle((left, z), right-left, layer.thickness_um,
                                  facecolor=colors[name], edgecolor='none'))
        z += layer.thickness_um
    if z == 0:
        ax.text(.5, .5, 'Uniform interface: no finite patterned layers', transform=ax.transAxes, ha='center', va='center')
    ax.set(xlim=(0., stack.pitch_um), ylim=(max(z, .1), 0.), xlabel='x (um)', ylabel='z into stack (um)',
           title=f'{stack.metadata.get("kind", "mark")} | pitch {stack.pitch_um:g} um | {len(stack.layers)} slices')
    ax.text(0., 1.015, f'Incident: {stack.top}; +z points downward', transform=ax.transAxes, fontsize=9)
    ax.text(0., -.22, f'Substrate: {stack.bottom}; shown geometry is the computed staircase', transform=ax.transAxes, fontsize=9)
    if names:
        ax.legend(handles=[Patch(facecolor=colors[name], label=name) for name in names], loc='upper right', fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
