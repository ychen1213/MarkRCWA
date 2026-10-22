"""Executed physical evidence, not self-reported unrun test success."""
import importlib.metadata
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


def run_validation(output='reports', run_pytest=True, independent=True):
    from .analytic import run_analytic_validation
    from .convergence import convergence_scan, write_convergence
    from .staircase import run_staircase_validation
    from mark_rcwa.cli import load_config, build, build_geometry
    directory = Path(output).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    records = run_analytic_validation()
    print(f'Analytic physical checks: {len(records)} executed',flush=True)
    if independent:
        from .independent import run_independent_validation
        records.extend(run_independent_validation())
        print('Independent solver checks executed',flush=True)
    else:
        records.append({'name':'independent solver comparison', 'status':'SKIP', 'reason':'explicitly disabled'})
    convergence_reports = []
    scenarios = [('rectangular_TE','configs/demo_symmetric.yaml','TE'),
                 ('rectangular_TM','configs/demo_symmetric.yaml','TM'),
                 ('asymmetric_TE','configs/demo_asymmetric.yaml','TE'),
                 ('asymmetric_TM','configs/demo_asymmetric.yaml','TM'),
                 ('multilayer_TM','configs/demo_multilayer.yaml','TM')]
    from dataclasses import replace
    for name,config_path,polarization in scenarios:
        if not Path(config_path).exists():
            records.append({'name':f'convergence/{name}', 'status':'SKIP', 'reason':f'missing repository example {config_path}'})
            continue
        config=load_config(config_path)
        stack,materials,spec=build(config)
        spec=replace(spec,polarization=polarization)
        reference=build_geometry(config['reference_mark'],materials) if 'reference_mark' in config else stack
        report=convergence_scan(stack,reference,materials,spec,config.get('convergence'))
        print(f'Convergence {name}: {report["status"]}, N={report["final_N"]}',flush=True)
        write_convergence(report,directory/'convergence'/name)
        convergence_reports.append({'name':name, 'report':report})
        records.append({'name':f'convergence/{name}', 'parameters':config,
            'solver':'Meent 0.13.2 scalar adapter, complex128',
            'expected':'two successive efficiency, individual/pair phase and APD changes meet thresholds',
            'actual':{'final_N':report['final_N'],'executed_orders':report['executed_orders'],
                      'phase_unreliable_orders':report['phase_unreliable_orders']},
            'error':{'final_pairs':report['history'][-1]['pairs']},
            'tolerances':report['tolerances'],
            'status':'PASS' if report['converged'] else 'FAIL',
            'reason':None if report['converged'] else 'truncation limit reached without consecutive phase/APD convergence'})
    records.extend(run_staircase_validation())
    print('Continuous-profile staircase refinement checks executed',flush=True)
    pytest_info = {'executed':False}
    if run_pytest and Path('tests').is_dir():
        xml_path=directory/'pytest.xml'
        command=[sys.executable,'-m','pytest','-q',f'--junitxml={xml_path}']
        completed=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (directory/'pytest.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
        pytest_info={'executed':True,'command':command,'returncode':completed.returncode}
        if xml_path.exists():
            test_counts=Counter()
            for testcase in ET.parse(xml_path).getroot().iter('testcase'):
                failure=testcase.find('failure')
                error=testcase.find('error')
                skipped=testcase.find('skipped')
                status='FAIL' if failure is not None or error is not None else 'SKIP' if skipped is not None else 'PASS'
                test_counts[status]+=1
                detail=failure if failure is not None else error if error is not None else skipped
                records.append({'name':'pytest/'+testcase.get('name'), 'parameters':{'suite':testcase.get('classname')},
                    'solver':'test assertions (details in source and pytest.log)',
                    'expected':'all assertions pass', 'actual':{'time_s':float(testcase.get('time','0'))},
                    'error':None if detail is None else detail.get('message'),
                    'status':status, 'reason':None if detail is None else detail.get('message')})
            pytest_info['counts']=dict(test_counts)
        else:
            records.append({'name':'pytest result log','status':'FAIL','reason':'pytest did not produce requested JUnit evidence'})
        if completed.returncode != 0:
            records.append({'name':'pytest execution', 'status':'FAIL', 'reason':f'exit {completed.returncode}; see pytest.log'})
    else:
        records.append({'name':'repository pytest', 'status':'SKIP','reason':'disabled or repository tests unavailable'})
    for name,reason in [('FDTD/FEM finite irregular mark benchmark','no local trusted solver installed'),
                        ('finite-beam stage-dependent scattering','not implemented or independently validated'),
                        ('RCWA/FDTD perturbative hybrid','not implemented or independently validated'),
                        ('2D / conical fields and GPU','outside validated scalar 1D CPU scope')]:
        records.append({'name':name,'status':'SKIP','reason':reason})
    counts=Counter(row['status'] for row in records)
    report={'timestamp_hk':datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat(),
        'solver':{'name':'Meent','version':importlib.metadata.version('meent'),
                  'commit':'bae28024476d4af2826b8412cb8288d79a0630ac','precision':'complex128'},
        'counts':dict(counts),'pytest':pytest_info,'tests':records,
        'convergence_reports':convergence_reports,
        'all_executed_required_checks_passed':counts['FAIL']==0,
        'limitations':'PASS applies only to recorded synthetic structures and discretizations; SKIP is not validation'}
    (directory/'validation_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    write_markdown_report(report,directory)
    return report


def write_markdown_report(report,directory):
    lines=['# Executed validation report','',f'Generated: {report["timestamp_hk"]}',
        '',f'Counts: {report["counts"]}. Repository pytest: {report["pytest"]}.',
        '', 'Every PASS below was executed. Detailed input structures, expected/actual values, errors and tolerances are in `validation_summary.json`.',
        '', '| Check | Status | Error / reason |', '| --- | --- | --- |']
    for row in report['tests']:
        if row['name'].startswith('pytest/') and row['status']=='PASS':
            continue
        detail=row.get('reason') or json.dumps(row.get('error'),ensure_ascii=False)
        if len(detail)>240:
            detail=detail[:237]+'... (full JSON)'
        lines.append(f'| {row["name"]} | {row["status"]} | {detail.replace("|","/")} |')
    lines += ['', '## Interpretation', '',
        'Energy alone does not establish phase convergence. Convergence requires two consecutive passing efficiency, amplitude-phase, pair-phase and APD changes; low-power pairs are explicitly excluded from phase accuracy. It is a numerical criterion, not a certified global error bound.',
        '', 'Analytic finite-film loss is checked independently against dissipated-power integration, rather than declaring R+T+(1-R-T)=1 as an independent absorption test.',
        '', 'S4 1.1.1 is an independent C++ implementation invoked through its Lua binary. The reference applies explicitly recorded tiny passive loss to internal lossless materials to stabilize its legacy eigensystem branch and checks the perturbation limit; exterior media remain unchanged. Main Meent inputs are not regularized. Installed Meent files are not patched.',
        '', 'Strictly lossless Meent native modal q rounding is corrected only in near-pure-imaginary propagating modes, with consistent V columns. TM lossy substrate epsilon is corrected in the adapter. Both fixes have independent regression evidence.',
        '', 'grcwa 0.1.2 TE comparison is recorded in `phase4_reference.json`; its Laurent TM at N55 did not reach 0.1 degree, and is not used to certify TM phase accuracy. Official grcwa Pol commit c52f10a reaches 0.1 degree on the researched rectangular case only at N155. Those historical probes are preserved in research notes.',
        '', 'The report does not validate experimental optical constants, any actual lithography instrument, finite-beam defect scanning, or RCWA/FDTD hybrid approximations. See `LIMITATIONS.md` and `INDEPENDENT_SOLVER.md`.']
    content='\n'.join(lines)+'\n'
    (directory/'VALIDATION_REPORT.md').write_text(content,encoding='utf-8')
    if Path('docs').is_dir():
        Path('docs/VALIDATION_REPORT.md').write_text(content,encoding='utf-8')
