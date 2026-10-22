from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_alignment(signals, apd, path):
    path = Path(path)
    count = len(signals)
    fig, axes = plt.subplots(count, 1, figsize=(10, max(3, 2.4*count)), squeeze=False, constrained_layout=True)
    for ax, (label, reference, test) in zip(axes[:,0], signals):
        ax.plot(reference.stage_um, reference.intensity, label='Reference', lw=1.3, alpha=.8)
        ax.plot(test.stage_um, test.intensity, label='Test', lw=1.1, alpha=.8)
        period = f'; period {test.period_um:.5g} um' if test.period_um is not None else ''
        ax.set(xlabel='Additional stage displacement (um)', ylabel='Detector intensity (a.u.)', title=label+period)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Idealized optical mode overlap; synthetic structures')
    fig.savefig(path/'stage_scans.png', dpi=160)
    plt.close(fig)
    valid = [row for row in apd if row['reliable']]
    fig, ax = plt.subplots(figsize=(7,4), constrained_layout=True)
    if valid:
        ax.bar([str(row['order']) for row in valid], [row['apd_nm'] for row in valid], color='#167d8d')
    else:
        ax.text(.5,.5,'No reliable collected +/- pair', transform=ax.transAxes, ha='center')
    ax.set(xlabel='Pair order m', ylabel='Model APD (nm)', title='APD from test-reference pair phase; branch limited')
    ax.axhline(0, color='black', lw=.7)
    fig.savefig(path/'apd.png', dpi=160)
    plt.close(fig)


def plot_morphology(records, parameter, path):
    fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
    orders = sorted({row['order'] for item in records for row in item['apd']})
    for m in orders:
        values = [next((row['apd_nm'] for row in item['apd'] if row['order']==m and row['reliable']), np.nan) for item in records]
        ax.plot([item['value'] for item in records], values, 'o-', label=f'+/-{m}')
    ax.set(xlabel=parameter, ylabel='Ideal-model APD (nm)', title='Synthetic morphology sweep; fixed reference mark')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(Path(path), dpi=160)
    plt.close(fig)
