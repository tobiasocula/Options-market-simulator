
from param_class import CrossExcitation
params = CrossExcitation(
    # ===== TIME SCALE =====
    dt=300.0,        # 5 minutes
    T=2340,         # ~30 trading days

    # ===== STATIC BASE INTENSITY =====
    alpha_moneyness=150.0,
    alpha_time=100.0,
    mu_intensity=0.005,    # was 0.04 → scaled down ~10×

    # ===== HAWKES DYNAMICS =====
    beta=0.3,           # was 0.3 → decay per 5-min step
    rho_self=1.0,
    tau=[[0.5, 0.4], [0.5, 0.4]],
    gamma_m=12.0,
    gamma_t=20.0,

    w_volume=0.5,

    # ===== VOLUME MODEL =====
    contract_volume_mean=0.6,
    contract_volume_std=0.7,
    volume_base=2.5,
    volume_moneyness=4.0,
    volume_time_decay=2.5,

    # ===== OPTION GRID =====
    strike_prices=[3800, 3900, 4000, 4100, 4200, 4300, 4400],
    expiry_dts=[86400*14, 86400*45, 86400*90, 86400*180],  # 2w, 1.5m, 3m, 6m

    # ===== FINANCE MODEL =====
    risk_free=0.04,
    dividend_rate=0.015,

    # ===== INITIALIZATION =====
    base_n_orders_init=10,
    base_scale_init_orders=0.01,
    moneyness_scale_init_orders=3.0,
    time_scale_init_orders=1.2,
    beta_init=0.1,
    gamma_init=0.5,

    init_open_price=4200.0,
    init_vola=0.04,

    # ===== HESTON =====
    kappa=1.8,
    theta=0.04,
    xi=0.25,
    mu=0.06,
    rho=-0.7,

    # ===== ORDER TYPE LOGIC =====
    limit_order_base_param = 1.2,
    limit_order_vol_param  = -0.3,
    limit_order_distance_param = 0.5,
    limit_order_spread_param = 0.5,

    buy_order_base_param = 0.0,
    buy_order_imbalance_param = 1.5
)