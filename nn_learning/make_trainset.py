import numpy as np
from param_class import CrossExcitation
from pathlib import Path
from debug import Debugger
from cross_excitation import cross_excitation

def sample_params(TUNABLE_PARAMS):

    p = {}
    
    # Scalars: multiplicative log-uniform around center
    p["mu_intensity"] = TUNABLE_PARAMS['mu_intensity']['center'] * np.exp(
        np.random.uniform(
            TUNABLE_PARAMS['mu_intensity']["rel_range"][0],
            TUNABLE_PARAMS['mu_intensity']["rel_range"][1]
        )
    )
    p['alpha_moneyness'] = TUNABLE_PARAMS["alpha_moneyness"]["center"] * np.exp(np.random.uniform(
        TUNABLE_PARAMS["alpha_moneyness"]["rel_range"][0],
        TUNABLE_PARAMS["alpha_moneyness"]["rel_range"][1]
        )
    )
    p['alpha_time'] = TUNABLE_PARAMS["alpha_time"]["center"] * np.exp(np.random.uniform(
        TUNABLE_PARAMS["alpha_time"]["rel_range"][0],
        TUNABLE_PARAMS["alpha_time"]["rel_range"][1]
        )
    )
    p['beta'] = TUNABLE_PARAMS["beta"]["center"] * np.exp(np.random.uniform(
        TUNABLE_PARAMS["beta"]["rel_range"][0],
        TUNABLE_PARAMS["beta"]["rel_range"][1]
        )
    )
    p['rho_self'] = TUNABLE_PARAMS["rho_self"]["center"] * np.random.uniform(
        TUNABLE_PARAMS["rho_self"]["rel_range"][0],
        TUNABLE_PARAMS["rho_self"]["rel_range"][1]
    )
    p['gamma_m'] = TUNABLE_PARAMS["gamma_m"]["center"] * np.random.uniform(
        TUNABLE_PARAMS["gamma_m"]["rel_range"][0],
        TUNABLE_PARAMS["gamma_m"]["rel_range"][1]
    )
    p['gamma_t'] = TUNABLE_PARAMS["gamma_t"]["center"] * np.random.uniform(
        TUNABLE_PARAMS["gamma_t"]["rel_range"][0],
        TUNABLE_PARAMS["gamma_t"]["rel_range"][1]
    )
    p['w_volume'] = TUNABLE_PARAMS["w_volume"]["center"] * np.random.uniform(
        TUNABLE_PARAMS["w_volume"]["rel_range"][0],
        TUNABLE_PARAMS["w_volume"]["rel_range"][1]
    )
    
    # tau matrix: perturb elementwise, then renormalize for stability
    tau_center = np.array(TUNABLE_PARAMS["tau"]["center"])
    tau_raw = tau_center * np.exp(np.random.uniform(
        TUNABLE_PARAMS["tau"]["rel_range"],
        TUNABLE_PARAMS["tau"]["rel_range"],
        size=(2,2))
    )
    tau_sum = tau_raw.sum()
    p['tau'] = (0.85 * tau_raw / tau_sum).tolist()  # branching ratio cap at 0.85
    
    # Quick stability check: reject if total excitation implausible
    return CrossExcitation(**p)

if __name__ == '__main__':

    TUNABLE_PARAMS = {
        'mu_intensity':      {'center': 0.015,  'rel_range': (0.3, 2.0)},    # baseline activity
        'alpha_moneyness':   {'center': 80.0,   'rel_range': (0.5, 1.5)},    # spatial width
        'alpha_time':        {'center': 30.0,   'rel_range': (0.5, 2.0)},    # maturity width  
        'beta':              {'center': 0.05,   'rel_range': (0.4, 2.0)},    # decay speed
        'rho_self':          {'center': 0.7,    'range':    (0.3, 0.95)},   # self-excitation
        'tau':               {'center': [[0.15,0.10],[0.15,0.10]], 'rel_range': (0.3, 1.2)}, # cross
        'gamma_m':           {'center': 4.0,    'rel_range': (0.5, 2.0)},    # strike kernel
        'gamma_t':           {'center': 5.0,    'rel_range': (0.5, 2.0)},    # expiry kernel  
        'w_volume':          {'center': 0.15,   'rel_range': (0.2, 2.0)}     # volume sensitivity
    }

    save_in = Path.cwd() / "nn_learning" / "training_data"

    num_samples = 10
    for k in range(num_samples):
        save_dir = save_in / f"set_{k}"
        save_dir.mkdir()

        paramset = sample_params(TUNABLE_PARAMS)

        dmode = 1
        debugger = Debugger(mode=dmode)
        cross_excitation(params=paramset, save=True, savedir=save_dir, debugger=debugger)