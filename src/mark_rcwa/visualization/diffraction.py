from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_diffraction(result, path, min_efficiency=1e-10):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    m = result.orders
    colors = np.where(result.collected, '#167d8d', np.where(result.propagating, '#e98b32', '#aaaaaa'))
    axs[0,0].bar(m, result.efficiency, color=colors)
    axs[0,0].set(ylabel='Reflected power / incident power', title=f'R={result.reflectance:.5f}; T={result.transmittance:.5f}')
    axs[0,1].plot(m, abs(result.amplitude), 'o-', ms=3)
    axs[0,1].set(ylabel='Reflected E amplitude magnitude', title='Complex amplitudes from RCWA')
    reliable = result.propagating & (result.efficiency >= min_efficiency)
    axs[1,0].plot(m[reliable], np.degrees(np.angle(result.amplitude[reliable])), 'o', ms=4)
    axs[1,0].set(ylabel='Phase (deg)', title='Low-power / evanescent phases omitted')
    p = result.propagating
    angles = np.degrees(np.arcsin(np.clip(result.kx[p].real/(result.n_top*2*np.pi/result.wavelength_um), -1, 1)))
    axs[1,1].scatter(angles, result.efficiency[p], c=colors[p], s=25)
    axs[1,1].set(xlabel='Far-field angle (deg)', ylabel='Reflected efficiency', title='Teal: collected; orange: outside pupil')
    for ax in axs.flat:
        ax.grid(alpha=.2)
    for ax in (axs[0,0], axs[0,1], axs[1,0]):
        ax.set_xlabel('Diffraction order m')
    fig.suptitle(f'{result.polarization}, {result.wavelength_um:g} um, incident {result.theta_deg:g} deg')
    fig.savefig(Path(path), dpi=160)
    plt.close(fig)


def plot_parameter_scan(scan, parameter, path):
    x = [item[0] for item in scan]
    fig, ax = plt.subplots(figsize=(7,4), constrained_layout=True)
    ax.plot(x, [r.reflectance for _,r in scan], 'o-', label='Total R')
    for m in (1,3,5):
        if all(m in r.orders for _,r in scan):
            ax.plot(x, [r.efficiency[r.orders == m][0] for _,r in scan], 'o--', label=f'R({m:+d})')
    ax.set(xlabel='Wavelength (um)' if parameter == 'wavelength_um' else 'Incident angle (deg)', ylabel='Diffraction efficiency')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(Path(path), dpi=160)
    plt.close(fig)
