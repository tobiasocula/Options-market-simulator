from cross_runs_scripts.params_long_2 import params
from pathlib import Path
from debug import Debugger
from cross_excitation import cross_excitation

"""
debug modes:
1: standard, minimal, shows excitations + num events
2: show volume + basics
3: detailed, shows kernel per contract
4: shows iv calculation
"""

results_dir = Path.cwd() / "cross_runs_results" / "long_2"
dmode = 3
debugger = Debugger(mode=dmode)
cross_excitation(params=params, save=True, savedir=results_dir, debugger=debugger)

"""
python cross_runs_scripts/run.py
"""
