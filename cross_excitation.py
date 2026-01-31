from param_class import CrossExcitation
import itertools
from scipy.stats import norm
import sys
from scipy.optimize import brentq
import numpy as np
from debug import Debugger

def cross_excitation(params: CrossExcitation, save=False, savedir=None,
                     debugger=None):
    
    if debugger is None:
        debugger = Debugger(None)

    # first part is exact same as self excitation

    def implied_vol_call(C_mkt, S, K, T, r, q, min_price=0.01):

        C_mkt = max(C_mkt, min_price)

        try:
            return brentq(
                lambda sigma: black_scholes_call(S, K, T, r, q, sigma)[0] - C_mkt,
                0.0, 10.0
            )
        except:
            return None

    def sample_multidim(probs):
        shape = probs.shape
        p_flat = probs.flatten()
        flat_idx = np.random.choice(probs.size, p=p_flat)
        indices = np.unravel_index(flat_idx, shape)
        return np.array(indices)
    
    def black_scholes_call(S, K, T, r, q, sigma):
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError(f"Invalid BS inputs: S={S}, K={K}, T={T}, sigma={sigma}")

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Option price
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        # Greeks
        delta = np.exp(-q * T) * norm.cdf(d1)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) * 0.01  # Scaled for 1% change in volatility
        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                r * K * np.exp(-r * T) * norm.cdf(d2) +
                q * S * np.exp(-q * T) * norm.cdf(d1)) / 365  # Per day
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01  # Scaled for 1% change in interest rate

        return call_price, delta, gamma, vega, theta, rho

    def black_scholes_put(S, K, T, r, q, sigma):
        (
            call_price, delta_call, gamma_call, vega_call, theta_call, rho_call
        ) = black_scholes_call(S, K, T, r, q, sigma)

        put_price = call_price - S * np.exp(-q * T) + K * np.exp(-r * T)

        delta_put = delta_call - np.exp(-q * T)
        gamma_put = gamma_call
        vega_put = vega_call
        theta_put = theta_call + r * K * np.exp(-r * T) - q * S * np.exp(-q * T)
        rho_put = rho_call - K * T * np.exp(-r * T)

        return put_price, delta_put, gamma_put, vega_put, theta_put, rho_put
        
    def rev(a):        
        n = len(a)
        return [a[n - k] for k in range(1, n+1)]

    def update_trades_ob(bids, asks, price, buy, vol, time, ltp, trades=None):

        if trades is None:
            trades = []

        if vol <= 0:
            return bids, asks, trades, ltp
        
        if buy:
            if asks is None or asks == [] or price < asks[0]["price"]:
                if bids is None or bids == []:
                    return [{"price": price, "volume": vol, "time": time}], asks, trades, ltp
                new_bids = rev(insert_bin(rev(bids), {"price": price, "volume": vol, "time": time}))
                return new_bids, asks, trades, ltp
            
            ask_order = asks[0]
            trade_vol = min(vol, ask_order["volume"])
            new_trades = trades + [{"price": ask_order["price"], "volume": trade_vol, "time": time}]
            
            if ask_order["volume"] > trade_vol:
                # Partial fill
                new_ask = insert_bin(asks[1:],
                            {"price": ask_order["price"], "volume": ask_order["volume"] - trade_vol, 
                            "time": ask_order["time"]})
                return update_trades_ob(bids, new_ask, price, buy, 0, time, ltp=ask_order["price"], trades=new_trades)  # vol=0 stops
            
            # Full/partial ask consumed, recurse with remainder
            return update_trades_ob(bids, asks[1:], price, buy, vol - ask_order["volume"], time=time, ltp=asks[0]["price"], trades=new_trades)
        
        else:  # Sell symmetric
            if bids is None or bids == [] or price > bids[0]["price"]:
                if asks is None or asks == []:
                    return bids, [{"price": price, "volume": vol, "time": time}], trades, ltp
                new_asks = insert_bin(asks, {"price": price, "volume": vol, "time": time})
                return bids, new_asks, trades, ltp
            
            bid_order = bids[0]
            trade_vol = min(vol, bid_order["volume"])
            new_trades = trades + [{"price": bid_order["price"], "volume": trade_vol, "time": time}]
            
            if bid_order["volume"] > trade_vol:
                new_bid = rev(insert_bin(rev(bids[1:]),
                            {"price": bid_order["price"], "volume": bid_order["volume"] - trade_vol, 
                            "time": bid_order["time"]}))
                return update_trades_ob(new_bid, asks, price, buy, 0, time, bid_order["price"], new_trades)
            
            return update_trades_ob(bids[1:], asks, price, buy, vol - bid_order["volume"], time, bids[0]["price"], new_trades)
        
    # sort item in already sorted array in O(log(n)) time
    def insert_bin(arr, item, field="price"):
        if arr == [] or arr is None:
            return [item]
        if arr[0][field] >= item[field]:
            return [item] + arr
        if arr[-1][field] <= item[field]:
            return arr + [item]
        
        L = 0
        N = len(arr) - 1
        U = N
        m = 0

        while U - L > 1:
            m = L + (U - L) // 2
            if arr[m][field] < item[field]:
                L = m
            elif arr[m][field] > item[field]:
                U = m
            else:
                break

        if arr[m][field] < item[field]:
            m += 1

        return arr[:m] + [item] + arr[m:]

    M = len(params.expiry_dts)
    N = len(params.strike_prices)
    T = params.T

    """
    keep track of statistics while running sim

    -orderbooks: raw orderbook data, for each contract (n, m, k), and for buy/sellside
    and every timestamp, keep track of current orderbook: meaning price, volume and time
    of order. orderbook gets updated each time a new trade happens (ie. order reduced/removed)

    -assetdata: for each timestamp, contains volatility and price value.

    -trades: for each contract (n, m, k) and each timestamp, keep track of all the trades
    that happened for this contract. the trades are cumulative (meaning a trade at time T
    could have happened between t=0 and t=T-1)

    -all_trades: keep track of every trade that happened over all contracts, for each
    timestamp. also cumulative. entries are dicts with fields the same for trades, but
    with extra fields including expiry, strike and type index.
    fields in order: price, time, volume, strike, expiry, type

    -kernels: for each contract (n, m, k) and timestamp, record the intensity of the kernel
    calculation. useful for tracking cross correlations between contracts and activity.

    -overviews: for each timestamp and contract, we keep an overview to track best bids,
    best asks, spreads, etc.

    -overviews_struct: different way of storing/formatting overviews. stores one dict
    per timestamp: dict has fields; strike, expiry, type, best_bid, best_ask, ...

    -buy_probs and limit_probs: for each contract and time, keep track of fraction
    of amount of buy vs sell orders and limit vs market orders.

    -num_events_all_contracts: keeps track of number of orders generated per contract and time.

    -intensities(keep): per contract and timestamp, record individual intensity contribution.

    -lambda(keep): per timestamp, record total intensity accross all contracts.

    -traded_volumes: traded volumes per contract and timestamp (NOT volumes from orders, but traded)

    """

    orderbooks = np.empty((M, N, 2, 2, T), dtype=object) # list of dicts
    assetdata = np.empty((2, T))
    trades = np.empty((M, N, 2, T), dtype=object) # list of dicts
    all_trades = []
    kernels = np.zeros((M, N, 2, T))
    overviews = np.empty((M, N, 2, T), dtype=object)
    overviews_struct = np.empty(T, dtype=object)
    intensities_keep = np.zeros((M, N, 2, T))
    num_events_keep = np.zeros(T)
    lambda_keep = np.zeros(T)
    num_events_all_contracts = np.zeros((M, N, 2, T))
    buys_probs = np.full((M, N, 2, T), np.nan)
    limit_probs = np.full((M, N, 2, T), np.nan)
    traded_volumes = np.zeros((M, N, 2, T))

    assetdata[0, 0] = params.init_open_price
    assetdata[1, 0] = params.init_vola

    C = N * M * 2 # normalization constant

    # initialize order books first (before running loop)
    for m, n, k in itertools.product(range(M), range(N), range(2)):
        time_till_expiry = params.expiry_dts[m] / (3600 * 24 * 365)
        fair_price, *_ = black_scholes_call(
            params.init_open_price,
            params.strike_prices[n],
            time_till_expiry,
            params.risk_free,
            params.dividend_rate,
            params.init_vola
        )
        moneyness = np.log(params.strike_prices[n] / params.init_open_price)
        
        # Liquidity scaling: fewer orders for OTM/long-dated
        rel_liq = np.exp(-params.gamma_init * moneyness - params.beta_init * time_till_expiry)
        n_orders_side = max(1, int(params.base_n_orders_init * rel_liq))  # ≥1 order/side
    
        # Price scale: wider spreads for OTM/long-dated
        scale_price = params.base_scale_init_orders * (
            1 + params.moneyness_scale_init_orders * moneyness + 
            params.time_scale_init_orders * time_till_expiry
        )

        # fill bid side first
        for _ in range(n_orders_side):
            offset = np.random.exponential(scale_price)  # Exp dist from fair
            order = {
                "price": max(fair_price - offset, 0.001),
                "volume": np.random.lognormal(params.contract_volume_mean, params.contract_volume_std),
                "time": 0.0
            }
            
            orderbooks[m, n, k, 0, 0] = insert_bin(
                orderbooks[m, n, k, 0, 0],
                order
            )

        # then fill ask side
        for _ in range(n_orders_side):
            offset = np.random.exponential(scale_price)  # Exp dist from fair
            order = {
                "price": max(0.01, fair_price + offset), # limit with some lower bound
                "volume": np.random.lognormal(params.contract_volume_mean, params.contract_volume_std),
                "time": 0.0
            }
            
            orderbooks[m, n, k, 1, 0] = insert_bin(
                orderbooks[m, n, k, 1, 0],
                order
            )

    dt_years = params.dt / (365 * 24 * 3600)

    excitations = np.zeros((M, N, 2))

    # begin main loop
    for T_current in range(1, T):
        debugger.debug(f"T current: {T_current}", mode=1)

        # step 1: update price
        z1 = np.random.normal()
        z2 = np.random.normal()
        dw_s = np.sqrt(dt_years) * z1
        dw_v = np.sqrt(dt_years) * (params.rho * z1 + np.sqrt(1 - params.rho**2) * z2)

        # update volatility
        assetdata[1, T_current] = assetdata[1, T_current - 1] + params.kappa * (params.theta - assetdata[1, T_current - 1]) * dt_years + params.xi * np.sqrt(assetdata[1, T_current - 1]) * dw_v
        # update price
        assetdata[0, T_current] = assetdata[0, T_current - 1] * np.exp((params.mu - 0.5 * assetdata[1, T_current - 1] **2) * dt_years + np.sqrt(assetdata[1, T_current - 1]) * dw_s)

        # step 2: generate potential new orders
        # do this for all contracts

        excitations *= np.exp(-params.beta * params.dt)  # decay past

        for this_exp, this_strike, this_type in itertools.product(range(M), range(N), range(2)):

            debugger.debug(f"Contract {this_exp}/{M-1}, {this_strike}/{N-1}, {this_type}/1", mode=3)

            time_till_expiry = abs(params.T * params.dt - params.expiry_dts[this_exp]) / (3600 * 24 * 365) # years
            moneyness = np.log(params.strike_prices[this_strike] / assetdata[0, T_current])
            adding_term = params.mu * np.exp(
                -params.alpha_moneyness * moneyness * moneyness
                -params.alpha_time * time_till_expiry
            )
            debugger.debug(f"adding term: {adding_term} with values {-params.alpha_moneyness * moneyness * moneyness} and {-params.alpha_time * time_till_expiry}", mode=3)

            excitations[this_exp, this_strike, this_type] += adding_term

            for other_exp, other_strike, other_type in itertools.product(range(M), range(N), range(2)):
                trade = trades[other_exp, other_strike, other_type, T_current - 1]
                if trade is None:
                    continue
                for entry in trade:
                    delta_t = params.gamma_t * np.abs(params.expiry_dts[this_exp] - params.expiry_dts[other_exp]) / (365 * 24 * 3600)
                    m_i = np.log(params.strike_prices[this_strike] / assetdata[0, T_current])
                    m_j = np.log(params.strike_prices[other_strike] / assetdata[0, T_current])
                    delta_m = params.gamma_m * np.abs(m_i - m_j)
                    first_part = (params.rho_self if (
                        this_exp == other_exp and this_strike == other_strike and this_type == other_type
                        ) else 0) + params.tau[this_type][other_type]
                    volume_part = params.w_volume * entry["volume"]
                    kernel = first_part * volume_part * np.exp(- delta_m - delta_t) / C

                    debugger.debug(f'volume part: {volume_part}', mode=2)
                    debugger.debug(f'delta_m: {delta_m}', mode=2)
                    debugger.debug(f'delta_t: {delta_t}', mode=2)
                    debugger.debug(f'kernel: {kernel}', mode=2)

                    kernels[this_exp, this_strike, this_type, T_current] = kernel
                    excitations[this_exp, this_strike, this_type] += kernel

                    debugger.debug(f"kernel: {kernel}", mode=3)

        print(excitations[:, :, 0])
        print()
        print(excitations[:, :, 1])
        Lambda = np.sum(excitations)
        debugger.debug(f"Lambda: {Lambda}", mode=1)

        lambda_keep[T_current] = Lambda

        num_events = np.random.poisson(Lambda * params.dt)
        num_events_keep[T_current] = num_events

        debugger.debug(f"num_events: {num_events}", mode=1)
        debugger.debug(f"num_events: {num_events}", mode=3)

        intensities_keep[:, :, :, T_current] = excitations[:, :, :]
        for _ in range(num_events):

            probs_per_contract = excitations / Lambda # same shape as intensities_prime
            chosen_idx = sample_multidim(probs_per_contract)
            chosen_exp, chosen_strike, chosen_type = chosen_idx

            num_events_all_contracts[chosen_exp, chosen_strike, chosen_type, T_current] += 1

            # generate order for this contract

            vol = int(np.random.lognormal(
                params.contract_volume_mean,
                params.contract_volume_std
            ) * params.volume_base * np.exp(
                -params.volume_moneyness * np.abs(np.log(
                    assetdata[0, T_current] / params.strike_prices[chosen_strike]
                )) - params.volume_time_decay * (
                    (params.expiry_dts[chosen_exp] - params.dt * T_current) / (3600 * 24 * 365)
            )))

            vol = max(1, vol) # round to nearest non-zero integer

            ob_bids = orderbooks[(chosen_exp, chosen_strike, chosen_type, 0, T_current - 1)]
            ob_asks = orderbooks[(chosen_exp, chosen_strike, chosen_type, 1, T_current - 1)]

            bid_vol_sum = sum([entry["volume"] for entry in ob_bids]) if ob_bids is not None else 0
            ask_vol_sum = sum([entry["volume"] for entry in ob_asks]) if ob_asks is not None else 0

            imbalance = (bid_vol_sum - ask_vol_sum) / (bid_vol_sum + ask_vol_sum + 1e-5)

            spread = abs(ob_bids[0]["price"] - ob_asks[0]["price"]) if (
                ob_asks != [] and ob_bids != []
                and
                ob_asks is not None and ob_bids is not None
            ) else 0.0

            eta_buy = params.buy_order_base_param + params.buy_order_imbalance_param * imbalance
            eta_limit = params.limit_order_base_param + params.limit_order_vol_param * vol + params.limit_order_spread_param * spread

            prob_buy = 1 / (1 + np.exp(-eta_buy))
            prob_limit = 1 / (1 + np.exp(-eta_limit))

            buy = np.random.uniform() <= prob_buy
            limit = np.random.uniform() <= prob_limit

            buys_probs[chosen_exp, chosen_strike, chosen_type, T_current] = prob_buy
            limit_probs[chosen_exp, chosen_strike, chosen_type, T_current] = prob_limit

            years_till_expiry = abs(params.expiry_dts[chosen_exp] - params.dt * T_current) / (3600 * 24 * 365)
            # determine price
            if chosen_type == 0:  # call
                theo, delta, gamma, vega, theta, rho = black_scholes_call(assetdata[0, T_current], params.strike_prices[chosen_strike], years_till_expiry,
                                        params.risk_free, params.dividend_rate,
                                        sigma=max(assetdata[1, T_current], 0.01) # cap to some lower bound
                )
            else:  # put
                theo, delta, gamma, vega, theta, rho = black_scholes_put(assetdata[0, T_current], params.strike_prices[chosen_strike], years_till_expiry,
                                        params.risk_free, params.dividend_rate,
                                        sigma=max(assetdata[1, T_current], 0.01) # cap to some lower bound
                )
                
            intrinsic = max(
                assetdata[0, T_current] - params.strike_prices[chosen_strike], 0
                ) if chosen_type == 0 else max(
                    params.strike_prices[chosen_strike] - assetdata[0, T_current], 0
                    )

            price = max(intrinsic, theo)
            
            if limit:
                if buy:
                    price += np.random.exponential(params.limit_order_distance_param * spread)
                else:
                    price -= np.random.exponential(params.limit_order_distance_param * spread)

            price = max(price, 0.01)

            if trades[chosen_exp, chosen_strike, chosen_type, T_current - 1] is None or trades[chosen_exp, chosen_strike, chosen_type, T_current - 1] == []:
                ltp = None
            else:
                ltp = trades[chosen_exp, chosen_strike, chosen_type, T_current - 1][-1]["price"]

            new_bids, new_asks, new_trades, new_ltp = update_trades_ob(
                bids=orderbooks[(chosen_exp, chosen_strike, chosen_type, 0, T_current - 1)],
                asks=orderbooks[(chosen_exp, chosen_strike, chosen_type, 1, T_current - 1)],
                price=price, buy=buy, vol=vol, time=params.dt * T_current, ltp=ltp
            )
            #print('new trades:'); print(new_trades)
            # Update global structures
            orderbooks[(chosen_exp, chosen_strike, chosen_type, 0, T_current)] = new_bids
            orderbooks[(chosen_exp, chosen_strike, chosen_type, 1, T_current)] = new_asks

            # update all trades structure
            for t in new_trades:
                # t is dict with (price, time, volume)
                # but we also need (expiry, strike, call/put)
                all_trades.append({
                    "price": t["price"],
                    "time": t["time"],
                    "volume": t["volume"],
                    "strike": params.strike_prices[chosen_strike],
                    "expiry": params.expiry_dts[chosen_exp],
                    "call/put": "call" if chosen_type == 0 else "put"
                })
                traded_volumes[chosen_exp, chosen_strike, chosen_type, T_current] += t["volume"]

            trades[chosen_exp, chosen_strike, chosen_type, T_current] = []
            # add new trades
            #print('adding new trades:', len(new_trades))
            for t in new_trades:
                trades[chosen_exp, chosen_strike, chosen_type, T_current].append(t)

            # update overview
            new_best_bid = None
            new_best_ask = None
            if new_bids is not None and new_bids != []:
                new_best_bid = new_bids[0]["price"]
            if new_asks is not None and new_asks != []:
                new_best_ask = new_asks[0]["price"]
            spread = new_best_ask - new_best_bid if (
                (new_bids is not None and new_bids != [])
                and
                (new_asks is not None and new_asks != [])
            ) else 0.0

            #prev_volume = overviews[chosen_exp, chosen_strike, chosen_type, T_current - 1]["volume"]
            new_added_vol = sum([t["volume"] for t in new_trades])

            overviews[chosen_exp, chosen_strike, chosen_type, T_current] = {
                "best_bid": new_best_bid if new_best_bid is not None else None,
                "best_ask": new_best_ask if new_best_ask is not None else None,
                "spread": spread,
                "ltp": new_ltp,
                "volume": new_added_vol,
                "moneyness": np.log(params.strike_prices[chosen_strike] / assetdata[0, T_current]),
                "iv": implied_vol_call(
                    C_mkt=(new_asks[0]["price"] + new_bids[0]["price"]) / 2,
                    S=assetdata[0, T_current],
                    K=params.strike_prices[chosen_strike],
                    T=years_till_expiry,
                    r=params.risk_free,
                    q=params.dividend_rate
                ) if (
                    (new_bids is not None and new_bids != [])
                    and
                    (new_asks is not None and new_asks != [])
                ) else None,
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "theta": theta,
                "rho": rho
            }

        # fill overviews_struct
        rows = []
        m, n, k = 0, 0, 0
        for idx in range(2 * M * N):
            if idx % N == 0 and idx != 0:
                m += 1
                n = 0
            if idx == N * M:
                k = 1
                n = 0
                m = 0

            this = overviews[m, n, k, T_current]
            if this is None:
                continue
            rows.append({
                "expiry": params.expiry_dts[m],
                "strike": params.strike_prices[n],
                "type": "call" if k == 0 else "put",
                "best_bid": this["best_bid"],
                "best_ask": this["best_ask"],
                "spread": this["spread"],
                "ltp": this["ltp"],
                "volume": this["volume"],
                "moneyness": this["moneyness"],
                "iv": this["iv"],
                "delta": this["delta"],
                "gamma": this["gamma"],
                "vega": this["vega"],
                "theta": this["theta"],
                "rho": this["rho"]
            })

            n += 1

        overviews_struct[T_current] = rows

    all_trades = np.array(all_trades, dtype=object)

    # end of loop

    if save:
        assert savedir is not None, AssertionError()
        np.save(savedir / "orderbooks.npy", orderbooks)
        np.save(savedir / "assetdata.npy", assetdata)
        np.save(savedir / "overviews.npy", overviews)
        np.save(savedir / "overviews_struct.npy", overviews_struct)
        np.save(savedir / "trades.npy", trades)
        np.save(savedir / "all_trades.npy", all_trades)
        np.save(savedir / "intensities_keep.npy", intensities_keep)
        np.save(savedir / "num_events.npy", num_events_keep)
        np.save(savedir / "lambda_keep.npy", lambda_keep)
        np.save(savedir / "limit_probs.npy", limit_probs)
        np.save(savedir / "buys_probs.npy", buys_probs)
        np.save(savedir / "num_events_contracts.npy", num_events_all_contracts)
        np.save(savedir / "kernels.npy", kernels)
        np.save(savedir / "traded_volumes.npy", traded_volumes)

    else:
        return (
            orderbooks,
            assetdata,
            overviews,
            overviews_struct,
            trades,
            all_trades,
            intensities_keep,
            lambda_keep,
            num_events_keep,
            limit_probs,
            buys_probs,
            num_events_all_contracts,
            kernels,
            traded_volumes
        )