
from param_class import CrossExcitation
params = CrossExcitation(
    # TIME SCALE
    dt=100_000,
    T=23,

    # STATIC BASE INTENSITY (per contract)
    alpha_moneyness = 0.1,
    alpha_time      =  0.0000003,
    mu_intensity    = 5e-6,
    mu_variation = 0.3,

    # HAWKES DYNAMICS
    beta     = 0.00005,           # slower decay over ~1–2 days
    rho_self = 0.7,            # self-excitation < 1
    tau      = [[0.15, 0.10],  # weaker cross-excitation
                [0.15, 0.10]],
    gamma_m  = 5.0,            # smoother in strikes
    gamma_t  = 0.0002,            # smoother in expiries

    w_volume = 0.1,           # lower volume impact

    # VOLUME MODEL (unchanged for now, tune after eyeballing data)
    contract_volume_mean = 2.0,
    contract_volume_std  = 1.2,
    volume_base          = 2.5,
    volume_moneyness     = 10.0,
    volume_time_decay    = 0.002,

    # OPTION GRID
    strike_prices = [3800, 3900, 4000, 4100, 4200, 4300, 4400],
    expiry_dts    = [86400 * k for k in [5, 15, 30, 45]],

    # FINANCE MODEL (leave as is)
    risk_free     = 0.04,
    dividend_rate = 0.015,

    # INITIALIZATION (as you have; revisit later)
    base_n_orders_init       = 10,
    base_scale_init_orders   = 0.01,
    moneyness_scale_init_orders = 3.0,
    time_scale_init_orders   = 1.2,
    beta_init                = 0.1,
    gamma_init               = 0.5,

    init_open_price = 120.0,
    init_vola       = 0.04,

    # HESTON
    kappa = 1.8,
    theta = 0.04,
    xi    = 0.25,
    mu    = 0.06,
    rho   = -0.7,

    # ORDER TYPE LOGIC (unchanged)
    limit_order_base_param    = 1.2,
    limit_order_vol_param     = -0.3,
    limit_order_distance_param= 0.5,
    limit_order_spread_param  = 0.5,

    buy_order_base_param      = 0.0,
    buy_order_imbalance_param = 1.5
)
