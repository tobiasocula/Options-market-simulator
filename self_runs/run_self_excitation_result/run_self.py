from self_runs.run_self_excitation_result.params_for_self_excitation import params
from self_excitation import self_excitation
from pathlib import Path

save = Path.cwd() / "run_self_excitation_result"
self_excitation(params, save=True, savedir=save)