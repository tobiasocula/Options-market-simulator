
from pydantic import BaseModel

class SelfExcitation(BaseModel):

    # generic
    dt: float # time increment (amount of time to pass per timestamp in simulation), in seconds
    T: int # amount of timesteps (no unit)

    # options trading
    mu_intensity: float # the static intensity per contract (also constant for all contracts here)
    # unit: events / second per contract

    beta: float # decay parameter, for this model, this is constant for all contracts k,j
    # unit: 1/second (decay rate per second)

    w_volume: float # determines strength of order volume
    alpha_moneyness: float # unit: none
    alpha_time: float # unit: 1/year

    # volume
    contract_volume_mean: float | int # mean for lognormal sampling of option contract size
    contract_volume_std: float | int # std for lognormal sampling of option contract size
    volume_base: float # for determining volume of contract orders (base parameter)
    volume_time_decay: float # for determining volume of contract orders (parameter for relation with time)
    volume_moneyness: float # for determining volume of contract orders (parameter for relation with moneyness of contract)

    # market params
    expiry_dts: list[int] # expiry dates (seconds from opening)
    strike_prices: list[float] # strike prices
    risk_free: float
    dividend_rate: float

    # orderbook init
    base_scale_init_orders: float # base price scale (exponential distribution for price; during construction of initial order book)
    moneyness_scale_init_orders: float # parameter for scaling price according to moneyness (init order book)
    time_scale_init_orders: float # parameter for scaling price according to time decay (init order book)
    base_n_orders_init: int # base number of orders per contract (init order book)
    beta_init: float # parameter for base liquidity calculation, with respect to time decay
    gamma_init: float # parameter for base liquidity calculation, with respect to moneyness

    # asset parameters
    init_open_price: float
    init_vola: float
    kappa: float # volatility mean reverting rate
    theta: float # volatility mean
    xi: float # volatility of volatility
    mu: float # asset yearly expected return
    rho: float # correlation volatility and asset price

    limit_order_base_param: float
    limit_order_vol_param: float
    limit_order_spread_param: float
    limit_order_distance_param: float

    buy_order_base_param: float
    buy_order_imbalance_param: float

class CrossExcitation(BaseModel):

    # options trading
    gamma_m: float # determines strength between contract's moneynesses
    gamma_t: float # determines strength between contract's expiry dates
    tau: list[list[float]] # cross-type intensity parameter
    rho_self: float # accounts for self-excitation
    mu_intensity: float # the static intensity per contract (also constant for all contracts here). unit: events/second
    beta: float # decay parameter, for this model, this is constant for all contracts k,j, unit: 1/second
    w_volume: float # determines strength of order volume, unit: 1/volume
    alpha_moneyness: float # unit: none
    alpha_time: float # unit: 1/second

    # generic
    dt: float # time increment (amount of time to pass per timestamp in simulation), seconds
    T: int # amount of timesteps

    # volume
    contract_volume_mean: float | int # mean for lognormal sampling of option contract size
    contract_volume_std: float | int # std for lognormal sampling of option contract size
    volume_base: float # for determining volume of contract orders (base parameter)
    volume_time_decay: float # for determining volume of contract orders (parameter for relation with time)
    volume_moneyness: float # for determining volume of contract orders (parameter for relation with moneyness of contract)

    # market params
    expiry_dts: list[int] # expiry dates (seconds from opening)
    strike_prices: list[float] # strike prices
    risk_free: float
    dividend_rate: float

    # orderbook init
    base_scale_init_orders: float # base price scale (exponential distribution for price; during construction of initial order book)
    moneyness_scale_init_orders: float # parameter for scaling price according to moneyness (init order book)
    time_scale_init_orders: float # parameter for scaling price according to time decay (init order book)
    base_n_orders_init: int # base number of orders per contract (init order book)
    beta_init: float # parameter for base liquidity calculation, with respect to time decay
    gamma_init: float # parameter for base liquidity calculation, with respect to moneyness

    # asset parameters
    init_open_price: float
    init_vola: float
    kappa: float # volatility mean reverting rate, unit: 1/year
    theta: float # volatility mean, unit: variance
    xi: float # volatility of volatility, unit: 1/sqrt(year)
    mu: float # asset yearly expected return, unit: 1/year
    rho: float # correlation volatility and asset price, unit: none

    limit_order_base_param: float
    limit_order_vol_param: float
    limit_order_spread_param: float
    limit_order_distance_param: float

    buy_order_base_param: float
    buy_order_imbalance_param: float