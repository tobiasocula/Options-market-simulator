
from param_class import CrossExcitation
params = CrossExcitation(

    # TIME SCALE
    dt=100_000,
    T=100,

    # BASE INTENSITY
    alpha_moneyness = 0.015,
    alpha_time      =   1e-05,
    mu_intensity    = 5e-5,
    mu_variation = 0.1,

    # HAWKES DYNAMICS
    beta     = 0.5,
    rho_self = 0.09, # must be < 1
    tau      = [[0.01, 0.005], [0.005, 0.01]],
    gamma_m  = 15.0,
    gamma_t  = 1e-04,

    w_volume = 0.08,

    # VOLUME MODEL
    contract_volume_mean = 2.0,
    contract_volume_std  = 0.5,
    volume_base          = 1.5,
    volume_moneyness     = 0.5,
    volume_time_decay    = 0.002,

    # OPTION GRID
    strike_prices = [120.0, 130.0, 140.0, 150.0],
    expiry_dts    = [86400 * k for k in [5, 15, 30, 45]], # num of seconds after open

    # FINANCE MODEL
    risk_free     = 0.04, # pct per year
    dividend_rate = 0.015, # pct per year

    # INITIALIZATION
    base_n_orders_init       = 10,
    base_scale_init_orders   = 0.01,
    moneyness_scale_init_orders = 3.0,
    time_scale_init_orders   = 1.2,
    beta_init                = 0.1,
    gamma_init               = 0.5,

    init_open_price = 130.0,
    init_vola       = 0.04,

    # HESTON
    kappa = 2.0,
    theta = 0.05,
    xi    = 0.3,
    mu    = 0.06,
    rho   = -0.5,

    # ORDER TYPE LOGIC
    limit_order_base_param    = 0.5,
    limit_order_vol_param     = -1.0,
    limit_order_distance_param= 0.5,
    limit_order_spread_param  = 0.5,

    buy_order_base_param      = 0.0,
    buy_order_imbalance_param = 1.5,
    )
