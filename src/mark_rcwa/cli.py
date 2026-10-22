import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import yaml
import numpy as np
from numbers import Integral
from mark_rcwa import geometry
from mark_rcwa.materials import from_config
from mark_rcwa.model import Slice, Stack
from mark_rcwa.solvers import MeentSolver, Simulation


def load_config(path):
    with open(path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or config.get('units') != 'um':
        raise ValueError('config must be a mapping with units: um')
    allowed={'units','description','geometry','materials','simulation','optics','reference_mark',
             'periodic_reference','detector','stage_scan','convergence','output','parameter_scans',
             'morphology_scan','diffraction_orders'}
    unknown=set(config)-allowed
    if unknown:
        raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
    sections={'optics':{'propagation_distance_um'},
        'stage_scan':{'start_um','stop_um','samples'},
        'detector':{'mode','order','orders','weights','channels','apd_orders','min_efficiency'},
        'morphology_scan':{'parameter','values'},
        'convergence':{'orders','monitored_orders','efficiency_tolerance','phase_tolerance_deg',
            'apd_tolerance_nm','min_efficiency','extend_orders','auto_extend'}}
    for name,keys in sections.items():
        if name in config:
            if not isinstance(config[name],dict):
                raise ValueError(f'{name} must be a mapping')
            extra=set(config[name])-keys
            if extra:
                raise ValueError(f'unknown {name} keys: {sorted(extra)}')
    for channel in config.get('detector',{}).get('channels',[]):
        if not isinstance(channel,dict):
            raise ValueError('detector channels require mappings with orders and weights')
        unknown=set(channel)-{'orders','weights'}
        if unknown:
            raise ValueError(f'unknown detector channel keys: {sorted(unknown)}')
    return config


def build_geometry(spec, materials):
    spec = dict(spec)
    kind = spec.pop('kind', 'rectangle')
    films = spec.pop('films', [])
    offsets = spec.pop('layer_offsets_um', {})
    cover = spec.pop('asymmetric_cover', None)
    if kind == 'layers':
        layer_specs=spec.pop('layers')
        if any(not isinstance(s,dict) or set(s)-{'thickness_um','edges_um','materials'} for s in layer_specs):
            raise ValueError('layers require mappings with only thickness_um, edges_um and materials')
        layers = tuple(Slice(float(s['thickness_um']), tuple(s['edges_um']), tuple(s['materials'])) for s in layer_specs)
        stack = Stack(layers=layers, **spec)
    elif kind == 'polygon':
        stack = geometry.from_polygon(materials=materials, **spec)
    elif kind == 'trapezoid':
        stack = geometry.trapezoid(**spec)
    elif kind == 'rectangle':
        stack = geometry.rectangle(**spec)
    elif kind == 'supercell':
        cells = [build_geometry(cell, materials) for cell in spec.pop('cells')]
        stack = geometry.supercell(cells, **spec)
    else:
        raise ValueError(f'unknown geometry kind: {kind}')
    for film in films[::-1]:
        stack = geometry.add_film(stack, **film)
    for index, offset in offsets.items():
        if isinstance(index,str):
            index=int(index)
        elif isinstance(index,bool) or not isinstance(index,Integral):
            raise ValueError('layer offset keys must be integer layer indices')
        stack = geometry.shift_layer(stack, index, offset)
    if cover:
        stack = geometry.asymmetric_cover(stack, **cover)
    for layer in stack.layers:
        for name in layer.materials:
            if name not in materials:
                raise ValueError(f'unknown material identifier: {name}')
    return stack


def build(config):
    materials = from_config(config.get('materials', {}))
    stack = build_geometry(config.get('geometry', {}), materials)
    sim = Simulation(**config.get('simulation', {}))
    return stack, materials, sim


def write_result(result, path):
    path.mkdir(parents=True, exist_ok=True)
    rows = result.rows()
    payload = dict(solver='Meent 0.13.2 + mark-rcwa 1D adapter',
        units='um', time_factor='exp(-i omega t)', polarization=result.polarization,
        pitch_um=result.pitch_um, wavelength_um=result.wavelength_um,
        reference_plane_um=result.reference_plane_um,
        lossless_q_branch_corrections=result.branch_corrections,
        cutoff_metadata=result.cutoff_metadata,
        R=result.reflectance, T=result.transmittance, A_flux_residual=result.absorption,
        orders=rows)
    (path/'diffraction.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    with (path/'diffraction.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def write_geometry(stack, materials, sim, path):
    from mark_rcwa.visualization.geometry import plot_geometry
    payload = dict(pitch_um=stack.pitch_um, original_pitch_um=stack.original_pitch_um,
        top=stack.top, bottom=stack.bottom, layers=[asdict(s) for s in stack.layers],
        metadata=stack.metadata,
        material_indices={name: {'n': materials.index(name, sim.wavelength_um).real,
            'k': materials.index(name, sim.wavelength_um).imag, 'source': materials[name].source}
            for name in sorted({stack.top, stack.bottom, *[m for s in stack.layers for m in s.materials]})})
    (path/'geometry.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    plot_geometry(stack, path/'geometry.png', materials=materials, wavelength_um=sim.wavelength_um)


def complex_weights(weights):
    def convert(value):
        if isinstance(value, (tuple,list)):
            if len(value) != 2:
                raise ValueError('complex transfer coefficient must be [real, imaginary]')
            return complex(*value)
        return complex(value)
    if weights is None:
        return None
    if isinstance(weights, dict):
        parsed={}
        for m,value in weights.items():
            if isinstance(m,str):
                m=int(m)
            elif isinstance(m,bool) or not isinstance(m,Integral):
                raise ValueError('optical weight keys must be integer diffraction orders')
            if int(m) in parsed:
                raise ValueError('duplicate optical weight order after integer key normalization')
            parsed[int(m)]=convert(value)
        return parsed
    return [convert(value) for value in weights]


def run_alignment(config, stack, materials, sim, result, path):
    from mark_rcwa.detector import scan, DetectorChannel
    from mark_rcwa.alignment import pair_apd
    from mark_rcwa.visualization.alignment import plot_alignment, plot_morphology
    reference_stack = build_geometry(config['reference_mark'], materials) if 'reference_mark' in config else stack
    reference = MeentSolver().solve(reference_stack, materials, sim)
    from mark_rcwa.propagation import propagate
    reference = propagate(reference, config.get('optics', {}).get('propagation_distance_um', 0))
    write_result(reference, path/'reference')
    write_geometry(reference_stack, materials, sim, path/'reference')
    stage = config.get('stage_scan', {})
    start, stop, samples = stage.get('start_um', -stack.pitch_um/2), stage.get('stop_um', stack.pitch_um/2), stage.get('samples', 2001)
    if not np.isfinite([start,stop]).all() or stop <= start or isinstance(samples,bool) or not isinstance(samples,int) or not 2 <= samples <= 1000000:
        raise ValueError('stage scan requires finite start<stop and integer samples between 2 and 1000000')
    x = np.linspace(start, stop, samples)
    detector = config.get('detector', {})
    mode = detector.get('mode', 'pair')
    weights = complex_weights(detector.get('weights'))
    min_eff = detector.get('min_efficiency', 1e-8)
    if mode=='pair':
        pair_orders=detector.get('orders',[detector['order']] if 'order' in detector else [1,3,5])
    else:
        if mode=='coherent':
            selected=detector.get('orders',[-5,-3,-1,1,3,5])
            default_pairs=sorted(m for m in selected if m>0 and -m in selected)
        elif mode=='independent':
            default_pairs=sorted({m for channel in detector.get('channels',[]) for m in channel['orders'] if m>0 and -m in channel['orders']})
        else:
            default_pairs=[]
        pair_orders=detector.get('apd_orders',default_pairs)
    if not isinstance(pair_orders,list) or (mode=='pair' and not pair_orders):
        raise ValueError('APD pair orders must be a list; pair mode requires at least one order')
    from mark_rcwa.detector import positive_pair_order
    pair_orders=[positive_pair_order(m) for m in pair_orders]
    if len(set(pair_orders))!=len(pair_orders):
        raise ValueError('APD pair orders must be unique')
    signals = []
    if mode == 'pair':
        if isinstance(weights,dict) and set(weights)!={q for m in pair_orders for q in (m,-m)}:
            raise ValueError('pair transfer weights must cover exactly all selected +/- orders')
        for m in pair_orders:
            pair_weights={q:weights[q] for q in (m,-m)} if isinstance(weights,dict) else weights
            signals.append((f'+/-{m} coherent pair', scan(reference,x,mode='pair',order=m,weights=pair_weights),
                scan(result,x,mode='pair',order=m,weights=pair_weights)))
    else:
        options = {'mode': mode, 'weights': weights}
        if mode == 'single':
            options['order'] = detector.get('order', 1)
        elif mode == 'coherent':
            options['orders'] = detector.get('orders', [-5,-3,-1,1,3,5])
        elif mode == 'independent':
            options['channels'] = [DetectorChannel(tuple(item['orders']), complex_weights(item.get('weights'))) for item in detector.get('channels', [])]
        signals.append((mode, scan(reference,x,**options), scan(result,x,**options)))
    apd = [asdict(pair_apd(reference, result, order=m, min_efficiency=min_eff)) for m in pair_orders]
    records = dict(model='idealized optical interference; no device calibration',
        computational_pitch_um=stack.pitch_um, original_pitch_um=stack.original_pitch_um,
        detector=detector, apd=apd,
        scan=[dict(label=label, metadata=test.metadata, stage_um=x.tolist(),
                   reference_intensity=ref.intensity.tolist(), test_intensity=test.intensity.tolist()) for label,ref,test in signals])
    (path/'alignment.json').write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
    with (path/'stage_scans.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['stage_um', *[f'{label}_{side}' for label,_,_ in signals for side in ('reference','test')]])
        for i,value in enumerate(x):
            writer.writerow([value, *[v for _,ref,test in signals for v in (ref.intensity[i],test.intensity[i])]])
    with (path/'apd.csv').open('w',newline='',encoding='utf-8') as stream:
        from dataclasses import fields
        from mark_rcwa.alignment import APDResult
        writer = csv.DictWriter(stream, fieldnames=list(apd[0]) if apd else [f.name for f in fields(APDResult)])
        writer.writeheader()
        writer.writerows(apd)
    plot_alignment(signals, apd, path)
    morph = config.get('morphology_scan')
    if morph:
        if morph['parameter'] not in ('left_angle_deg','right_angle_deg','duty','depth_um','top_radius_um','bottom_radius_um'):
            raise ValueError('unsupported morphology parameter')
        sweep = []
        for value in morph['values']:
            geometry_spec = {**config.get('geometry', {}), morph['parameter']: value}
            item = MeentSolver().solve(build_geometry(geometry_spec, materials), materials, sim)
            item = propagate(item, config.get('optics', {}).get('propagation_distance_um', 0))
            sweep.append(dict(value=value, apd=[asdict(pair_apd(reference, item, order=m, min_efficiency=min_eff)) for m in pair_orders]))
        (path/'morphology_scan.json').write_text(json.dumps(sweep, indent=2, allow_nan=False), encoding='utf-8')
        plot_morphology(sweep, morph['parameter'], path/'morphology_scan.png')
    return apd


def write_selected_orders(config,result,path):
    if 'diffraction_orders' not in config:
        return
    orders=config['diffraction_orders']
    if not isinstance(orders,list) or not orders or any(isinstance(m,bool) or not isinstance(m,int) for m in orders) or len(set(orders))!=len(orders):
        raise ValueError('diffraction_orders must be a nonempty unique integer list')
    rows={row['order']:row for row in result.rows()}
    if any(m not in rows for m in orders):
        raise ValueError('requested diffraction_orders outside retained Fourier basis')
    selected=[rows[m] for m in orders]
    (path/'selected_orders.json').write_text(json.dumps(selected,indent=2,allow_nan=False),encoding='utf-8')
    with (path/'selected_orders.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=selected[0])
        writer.writeheader()
        writer.writerows(selected)


def run_supercell_comparison(config, stack, materials, sim, result, path):
    from dataclasses import replace
    from mark_rcwa.visualization.diffraction import plot_diffraction
    import matplotlib.pyplot as plt
    mapping = geometry.order_mapping(stack, result.orders)
    (path/'supercell_mapping.json').write_text(json.dumps(mapping, indent=2, allow_nan=False), encoding='utf-8')
    with (path/'supercell_mapping.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=mapping[0])
        writer.writeheader()
        writer.writerows(mapping)
    if 'periodic_reference' not in config:
        return
    base = build_geometry(config['periodic_reference'], materials)
    count = round(stack.pitch_um/stack.original_pitch_um)
    base_sim = replace(sim, fourier_order=sim.fourier_order//count)
    base_result = MeentSolver().solve(base, materials, base_sim)
    subpath = path/'periodic_reference'
    write_result(base_result, subpath)
    write_geometry(base, materials, base_sim, subpath)
    plot_diffraction(base_result, subpath/'diffraction.png')
    fig, ax = plt.subplots(figsize=(10,4.5), constrained_layout=True)
    ax.stem(base_result.orders, base_result.efficiency, linefmt='C1-', markerfmt='C1o', basefmt=' ', label='Single original periodic cell')
    ax.scatter(result.orders/count, result.efficiency, s=15, label=f'Irregular {count}-cell periodic supercell', color='#167d8d')
    ax.set(xlabel='Grating wavevector / (2 pi / original pitch)', ylabel='Reflected diffraction efficiency',
           title='Fractional orders are supercell scattering; supercell repeats indefinitely')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(path/'supercell_comparison.png', dpi=160)
    plt.close(fig)
    (path/'supercell_comparison.json').write_text(json.dumps(dict(
        original_pitch_um=base.pitch_um, supercell_pitch_um=stack.pitch_um,
        original_N=base_sim.fourier_order, supercell_N=sim.fourier_order,
        reciprocal_cutoff_same=base_sim.fourier_order*count == sim.fourier_order,
        reference_R=base_result.reflectance, supercell_R=result.reflectance,
        interpretation='Different physical structures; repeated supercell is not an isolated finite mark'), indent=2), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='mark-rcwa')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('simulate', 'alignment'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True)
        command.add_argument('--output')
    converge = commands.add_parser('converge')
    converge.add_argument('--config', required=True)
    converge.add_argument('--output')
    validate = commands.add_parser('validate')
    validate.add_argument('--output',default='reports')
    validate.add_argument('--skip-independent',action='store_true')
    validate.add_argument('--skip-pytest',action='store_true')
    benchmark=commands.add_parser('benchmark')
    benchmark.add_argument('--output',default='reports/benchmark')
    args = parser.parse_args(argv)
    try:
        if args.command == 'benchmark':
            from mark_rcwa.benchmark import run_benchmark
            result=run_benchmark(args.output)
            print(f'Benchmark: {len(result["scenarios"])} CPU scenarios; {Path(args.output).resolve()}')
            return 0
        if args.command == 'validate':
            from mark_rcwa.validation import run_validation
            report=run_validation(args.output,not args.skip_pytest,not args.skip_independent)
            print(f'Validation: {report["counts"]}; {Path(args.output).resolve()}')
            return 0 if report['all_executed_required_checks_passed'] else 1
        config = load_config(args.config)
        stack, materials, sim = build(config)
        if args.command == 'converge':
            from mark_rcwa.validation.convergence import convergence_scan, write_convergence
            reference_stack = build_geometry(config['reference_mark'],materials) if 'reference_mark' in config else stack
            report = convergence_scan(stack,reference_stack,materials,sim,config.get('convergence'))
            path = Path(args.output or config.get('output','outputs/convergence'))
            write_convergence(report,path)
            print(f'{report["status"]}: N={report["final_N"]}; {path.resolve()}')
            return 0 if report['converged'] else 1
        result = MeentSolver().solve(stack, materials, sim)
        from mark_rcwa.propagation import propagate
        result = propagate(result, config.get('optics', {}).get('propagation_distance_um', 0))
        path = Path(args.output or config.get('output', 'outputs/simulation'))
        write_result(result, path)
        write_selected_orders(config,result,path)
        write_geometry(stack, materials, sim, path)
        from mark_rcwa.visualization.diffraction import plot_diffraction, plot_parameter_scan
        from mark_rcwa.diffraction import parameter_scan
        plot_diffraction(result, path/'diffraction.png')
        for parameter, values in config.get('parameter_scans', {}).items():
            scan = parameter_scan(stack, materials, sim, parameter, values)
            scan = [(value,propagate(r,config.get('optics',{}).get('propagation_distance_um',0))) for value,r in scan]
            records = [{'value': value, 'R': r.reflectance, 'T': r.transmittance,
                'reference_plane_um':r.reference_plane_um, 'orders': r.rows()} for value,r in scan]
            (path/f'scan_{parameter}.json').write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
            plot_parameter_scan(scan, parameter, path/f'scan_{parameter}.png')
        (path/'input_config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf-8')
        if args.command == 'alignment':
            apd = run_alignment(config, stack, materials, sim, result, path)
            if apd:
                print('APD (nm): '+', '.join(f'{row["order"]}: {row["apd_nm"]}' for row in apd))
        if stack.original_pitch_um is not None:
            run_supercell_comparison(config, stack, materials, sim, result, path)
        print(f'R={result.reflectance:.9f}, T={result.transmittance:.9f}, A={result.absorption:.9f}; {path.resolve()}')
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')
