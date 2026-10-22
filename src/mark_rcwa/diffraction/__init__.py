from dataclasses import replace
from mark_rcwa.solvers import MeentSolver


def parameter_scan(stack, materials, simulation, parameter, values):
    if parameter not in ('wavelength_um', 'theta_deg'):
        raise ValueError('parameter scan supports wavelength_um or theta_deg')
    solver = MeentSolver()
    return [(float(value), solver.solve(stack, materials, replace(simulation, **{parameter: float(value)})))
            for value in values]
