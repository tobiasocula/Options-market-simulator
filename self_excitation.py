
import numpy as np
from param_class import SelfExcitation
import itertools
from scipy.stats import norm
import sys
from scipy.optimize import brentq

def self_excitation(params: SelfExcitation, save=False, savedir=None):
    
    def implied_vol_call(C_mkt, S, K, T, r, q, min_price=0.01, max_tries=3):

        C_mkt = max(C_mkt, min_price)
        
        # def bs_diff(sigma):
        #     try:
        #         return black_scholes_call(S, K, T, r, q, sigma) - C_mkt
        #     except:
        #         #raise AssertionError(f"failed to call black scholes with params {C_mkt}, {S}, {K}, {T}")
        #         return None

        try:
            return brentq(
                lambda sigma: black_scholes_call(S, K, T, r, q, sigma) - C_mkt,
                0.0, 10.0
            )
        except:
            return None
            
        
        # # Adaptive bounds
        # low = 1e-4
        # high = 5.0
        
        # for _ in range(max_tries):
        #     f_low = bs_diff(low)
        #     f_high = bs_diff(high)
        #     if np.sign(f_low) != np.sign(f_high):
        #         return brentq(bs_diff, low, high)
        #     high *= 2  # Expand upper bound
        
        # # solver failed, return None
        # #raise AssertionError(f"failed to call black scholes with params {C_mkt}, {S}, {K}, {T}")
        # return None

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

        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def black_scholes_put(S, K, T, r, q, sigma):
        call = black_scholes_call(S, K, T, r, q, sigma)
        return call - S * np.exp(-q * T) + K * np.exp(-r * T)
    
    def rev(a):        
        n = len(a)
        return [a[n - k] for k in range(1, n+1)]

    def update_trades_ob(bids, asks, price, buy, vol, time, ltp, trades=None):

        if vol <= 0:
            return bids, asks, trades, ltp

        if trades is None:
            trades = []
        
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

    orderbooks = np.empty((M, N, 2, 2, T), dtype=object) # list of dicts
    assetdata = np.empty((2, T))
    trades = np.empty((M, N, 2, T), dtype=object) # list of dicts
    all_trades = np.empty(T, dtype=object) # list of dicts, with fields
    # price, time, volume, expiry, strike, call/put
    overviews = np.empty((M, N, 2, T), dtype=object) # list of dicts with fields:
    # best bid, best ask, spread, volume, LTP, moneyness, IV

    assetdata[0, 0] = params.init_open_price
    assetdata[1, 0] = params.init_vola

    intensities_keep = np.zeros((M, N, 2, T))
    num_events_keep = np.zeros(T)
    lambda_keep = np.zeros(T)
    buys_probs = np.empty((M, N, 2, T))
    limit_probs = np.empty((M, N, 2, T))

    # initialize order books first (before running loop)
    for m, n, k in itertools.product(range(M), range(N), range(2)):
        time_till_expiry = params.expiry_dts[m] / (3600 * 24 * 365)
        fair_price = black_scholes_call(
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
                "price": fair_price + offset,
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

        # step 1: update price
        z1 = np.random.normal()
        z2 = np.random.normal()
        dw_s = np.sqrt(dt_years) * z1
        dw_v = np.sqrt(dt_years) * (params.rho * z1 + np.sqrt(1 - params.rho**2) * z2)

        # update volatility
        assetdata[1, T_current] = assetdata[1, T_current - 1] + params.kappa * (params.theta - assetdata[1, T_current - 1]) * dt_years + params.xi * np.sqrt(assetdata[1, T_current - 1]) * dw_v
        # update price
        assetdata[0, T_current] = assetdata[0, T_current - 1] * np.exp((params.mu - 0.5 * assetdata[1, T_current - 1] **2) * dt_years + np.sqrt(assetdata[0, T_current - 1]) * dw_s)

        # step 2: generate potential new orders
        # do this for all contracts

        print('excitations before:')
        print(excitations[:, :, 0])
        print()

        excitations *= np.exp(-params.beta * params.dt)  # decay past
        print('after 1')
        print(excitations[:, :, 0])
        excitations += params.mu_intensity             # add baseline
        print('after 1')
        print(excitations[:, :, 0])

        for this_exp, this_strike, this_type in itertools.product(range(M), range(N), range(2)):
            this_trades = trades[this_exp, this_strike, this_type, T_current]

            # account for moneyness and expiry time
            time_till_expiry = (params.T * params.dt - params.expiry_dts[this_exp]) / (3600 * 24 * 365) # years
            print('time till expiry:', time_till_expiry)
            moneyness = np.log(params.strike_prices[this_strike] / assetdata[0, T_current])
            excitations[this_exp, this_strike, this_type] += np.exp(
                -params.alpha_moneyness * moneyness * moneyness
                -params.alpha_time * time_till_expiry
            )

            if this_trades is None or this_trades == []:
                continue
            print('ADDING', len(this_trades), 'MANY TRADES')
            for entry in this_trades:
                # entry is dict with fields: price, time, volume
                kernel = np.exp(
                    params.w_volume * np.log(entry["volume"] + 1e-8)
                )
                excitations[this_exp, this_strike, this_type, T_current] += kernel


        Lambda = np.sum(excitations)

        lambda_keep[T_current] = Lambda

        num_events = np.random.poisson(Lambda * params.dt)
        num_events_keep[T_current] = num_events

        intensities_keep[:, :, :, T_current] = excitations[:, :, :]

        print('num events:', num_events)
        print('lambda:', Lambda)

        for _ in range(num_events):

            probs_per_contract = excitations / Lambda # same shape as intensities_prime
            chosen_idx = sample_multidim(probs_per_contract)
            chosen_exp, chosen_strike, chosen_type = chosen_idx

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

            eta_buy = params.buy_order_base_param + params.buy_order_vol_param * vol
            eta_limit = params.limit_order_base_param + params.limit_order_vol_param * vol

            prob_buy = 1 / (1 + np.exp(-eta_buy))
            prob_limit = 1 / (1 + np.exp(-eta_limit))

            buy = np.random.uniform() <= prob_buy
            limit = np.random.uniform() <= prob_limit

            buys_probs[chosen_exp, chosen_strike, chosen_type, T_current] = prob_buy
            limit_probs[chosen_exp, chosen_strike, chosen_type, T_current] = prob_limit

            years_till_expiry = (params.expiry_dts[chosen_exp] - params.dt * T_current) / (3600 * 24 * 365)
            # determine price
            if chosen_type == 0:  # call
                theo = black_scholes_call(assetdata[0, T_current], params.strike_prices[chosen_strike], years_till_expiry,
                                        params.risk_free, params.dividend_rate,
                                        params.init_vola)
            else:  # put
                theo = black_scholes_put(assetdata[0, T_current], params.strike_prices[chosen_strike], years_till_expiry,
                                        params.risk_free, params.dividend_rate,
                                        params.init_vola)

            price = max(theo, theo*0.95 + np.random.exponential(0.001))  # Floor + noise
            if limit:
                price += np.random.exponential(params.limit_order_distance_param)

            if trades[chosen_exp, chosen_strike, chosen_type, T_current - 1] is None or trades[chosen_exp, chosen_strike, chosen_type, T_current - 1] == []:
                ltp = None
            else:
                ltp = trades[chosen_exp, chosen_strike, chosen_type, T_current - 1][-1]["price"]

            new_bids, new_asks, new_trades, new_ltp = update_trades_ob(
                bids=orderbooks[(chosen_exp, chosen_strike, chosen_type, 0, T_current - 1)],
                asks=orderbooks[(chosen_exp, chosen_strike, chosen_type, 1, T_current - 1)],
                price=price, buy=buy, vol=vol, time=params.dt * T_current, ltp=ltp
            )
            # Update global structures
            orderbooks[(chosen_exp, chosen_strike, chosen_type, 0, T_current)] = new_bids
            orderbooks[(chosen_exp, chosen_strike, chosen_type, 1, T_current)] = new_asks

            # update all trades structure
            if all_trades[T_current - 1] is None:
                all_trades[T_current] = []
            else:
                all_trades[T_current] = all_trades[T_current - 1]

            for t in new_trades:
                # t is dict with (price, time, volume)
                # but we also need (expiry, strike, call/put)
                all_trades[T_current].append({
                    "price": t["price"],
                    "time": t["time"],
                    "volume": t["volume"],
                    "strike": params.strike_prices[chosen_strike],
                    "expiry": params.expiry_dts[chosen_exp],
                    "call/put": "call" if chosen_type == 0 else "put"
                })

            # copy trades from previous timestamp
            if trades[chosen_exp, chosen_strike, chosen_type, T_current - 1] is None:
                trades[chosen_exp, chosen_strike, chosen_type, T_current] = []
            else:
                trades[chosen_exp, chosen_strike, chosen_type, T_current] = trades[chosen_exp, chosen_strike, chosen_type, T_current - 1][:]

            # add new trades
            for t in new_trades:
                trades[chosen_exp, chosen_strike, chosen_type, T_current].append(t)

            # update overview
            if new_bids is not None and new_bids != []:
                new_best_bid = new_bids[0]["price"]
            if new_asks is not None and new_asks != []:
                new_best_ask = new_asks[0]["price"]
            spread = new_best_ask - new_best_bid if (
                (new_bids is not None and new_bids != [])
                and
                (new_asks is not None and new_asks != [])
            ) else 0.0  

            overviews[chosen_exp, chosen_strike, chosen_type, T_current] = {
                "best_bid": new_best_bid if new_best_bid is not None else None,
                "best_ask": new_best_ask if new_best_ask is not None else None,
                "spread": spread,
                "ltp": new_ltp,
                "volume": 0.0, # to change
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
                ) else None
            }

    # compute total buy and limit order probabilities (over time)
    buys_probs_mean = np.mean(buys_probs, axis=-1)
    limit_probs_mean = np.mean(limit_probs, axis=-1)

    if save:
        assert savedir is not None, AssertionError()
        np.save(savedir / "orderbooks.npy", orderbooks)
        np.save(savedir / "assetdata.npy", assetdata)
        np.save(savedir / "overviews.npy", overviews)
        np.save(savedir / "trades.npy", trades)
        np.save(savedir / "all_trades.npy", all_trades)
        np.save(savedir / "intensities_keep.npy", intensities_keep)
        np.save(savedir / "num_events.npy", num_events_keep)
        np.save(savedir / "lambda_keep.npy", lambda_keep)
        np.save(savedir / "buys_probs.npy", buys_probs)
        np.save(savedir / "limit_probs.npy", limit_probs)
        np.save(savedir / "buys_probs_mean.npy", buys_probs_mean)
        np.save(savedir / "limit_probs_mean.npy", limit_probs_mean)

    else:
        return (
            orderbooks,
            assetdata,
            overviews,
            trades,
            all_trades,
            intensities_keep,
            lambda_keep,
            num_events_keep,
            buys_probs,
            limit_probs,
            buys_probs_mean,
            limit_probs_mean
        )
    




