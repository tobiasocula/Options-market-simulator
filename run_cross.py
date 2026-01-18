from params_for_cross_excitation import params
from cross_excitation import cross_excitation
from pathlib import Path

save = Path.cwd() / "run_cross_excitation_result"
cross_excitation(params, save=True, savedir=save)