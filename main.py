import numpy as np
import yfinance as yf
import pandas as pd
import itertools
import copy
import matplotlib.pyplot as plt

# ==============================
#           CONFIG
# ==============================
MAX_HEDGE = 0.2

HEDGE_UNIVERSE = [
    {"ticker": "SH", "type": "inverse", "max_weight": 0.4},
    {"ticker": "TLT", "type": "bond", "max_weight": 0.6},
    {"ticker": "GLD", "type": "gold", "max_weight": 0.5},
]

PORTFOLIO = [
    {"ticker": "BTC-USD", "amount": 1000.0},
    {"ticker": "NVDA", "amount": 500.0},
    {"ticker": "TSLA", "amount": 1200.0},
    {"ticker": "AAPL", "amount": 900.0},
]

# ==============================
#     HEDGE COMB UTILITIES
# ==============================
def get_max_weight(ticker: str, hedge_universe: list[dict]) -> float:
    for h in hedge_universe:
        if h["ticker"] == ticker:
            return h["max_weight"]
    raise ValueError(f"[get_max_weight] {ticker} not found")


def generate_hedge_combs(hedge_universe, max_hedge: float):
    hedge_combs = [{"name": "BASE", "weights": {}}]

    tickers = [h["ticker"] for h in hedge_universe]
    n = len(tickers)

    for k in range(1, n + 1):
        for combo in itertools.combinations(tickers, k):
            w_each = max_hedge / k
            valid = True

            for t in combo:
                if w_each > get_max_weight(t, hedge_universe):
                    valid = False
                    break
            if not valid:
                continue

            weights = {t: w_each for t in combo}
            name = "+".join(f"{t}_{int(w_each * 100)}%" for t in combo)

            hedge_combs.append({"name": name, "weights": weights})

    return hedge_combs


# ==============================
#     PORTFOLIO UTILITIES
# ==============================
def build_base_and_weighted_portfolio(portfolio, max_hedge):
    pf_amounts = np.array([pf["amount"] for pf in portfolio])

    base_weights = pf_amounts / pf_amounts.sum()
    scaled_pf_weights = pf_amounts / (pf_amounts.sum() / (1 - max_hedge))

    base_portfolio = copy.deepcopy(portfolio)
    for i in range(len(base_portfolio)):
        base_portfolio[i]["amount"] = base_weights[i]

    weighted_portfolio = copy.deepcopy(portfolio)
    for i in range(len(weighted_portfolio)):
        weighted_portfolio[i]["amount"] = scaled_pf_weights[i]

    return base_portfolio, weighted_portfolio


def build_hedged_portfolios(base_portfolio, weighted_portfolio, hedge_combs):
    hedged = []

    hedged.append({
        "name": "BASE",
        "weights": {p["ticker"]: p["amount"] for p in base_portfolio}
    })

    for comb in hedge_combs[1:]:
        pf_w = {p["ticker"]: p["amount"] for p in weighted_portfolio}
        for t, w in comb["weights"].items():
            pf_w[t] = w

        hedged.append({
            "name": comb["name"],
            "weights": pf_w,
        })

    return hedged


# ==============================
#       RETURNS & METRICS
# ==============================
def download(tickers, start, end):
    return yf.download(tickers, start=start, end=end)["Close"]


def to_log(prices):
    return np.log(prices / prices.shift(1)).dropna()


def evaluate_hedged_portfolios(hedged_portfolio, start, end):
    for pf in hedged_portfolio:
        tickers = list(pf["weights"].keys())
        weights = np.array(list(pf["weights"].values()))

        prices = download(tickers, start, end)
        log_rets = to_log(prices)

        pf_log_ret = (log_rets * weights).sum(axis=1)
        n = len(pf_log_ret)

        pf_cumsum = pf_log_ret.cumsum()
        pf_CAGR = np.exp(pf_cumsum.iloc[-1] * 252 / n) - 1

        peak = pf_cumsum.cummax()
        dd = pf_cumsum - peak
        pf_maxDD = (np.exp(dd) - 1).min()

        pf["cagr"] = pf_CAGR
        pf["maxdd"] = pf_maxDD


def score_hedges(hedged_portfolio):
    base = hedged_portfolio[0]

    dd_gain = []
    cagr_loss = []
    scores = []
    eps = 1e-6

    for pf in hedged_portfolio[1:]:
        d_dd = abs(base["maxdd"]) - abs(pf["maxdd"])
        dd_gain.append(d_dd)

        d_cagr = base["cagr"] - pf["cagr"]
        cagr_loss.append(d_cagr)

        if d_dd <= 0:
            s = -np.inf
        elif d_cagr <= 0:
            s = 1e9
        else:
            s = d_dd / (d_cagr + eps)

        scores.append(s)

    for i, pf in enumerate(hedged_portfolio[1:]):
        pf["score"] = scores[i]

    hedged_portfolio[0]["score"] = np.nan

    return dd_gain, cagr_loss, scores


# ==============================
#     REPORTING UTILITIES
# ==============================
def summarize_hedged_portfolios(hedged_portfolio):
    rows = []
    for pf in hedged_portfolio:
        rows.append({
            "name": pf["name"],
            "cagr": pf["cagr"],
            "maxdd": pf["maxdd"],
            "score": pf["score"],
        })
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    return df


def pretty_print_result(result):
    best = result["best"]

    print("===== Best Hedge Strategy =====")
    print(f"Name   : {best['name']}")
    print(f"Score  : {best['score']:.4f}")
    print(f"DD Gain: {best['dd_gain']:.4%}")
    print(f"CAGR Δ : {best['cagr_loss']:.4%}")
    print("Weights:")
    for t, w in best["weights"].items():
        print(f"  - {t}: {w:.4f}")

    print("\n===== All Strategies =====")
    df = summarize_hedged_portfolios(result["hedged_portfolio"])
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))


# ==============================
#          PLOTTING
# ==============================
def build_cumsum_portfolio(weights_dict, start, end):
    tickers = list(weights_dict.keys())
    weights = np.array(list(weights_dict.values()))

    prices = download(tickers, start, end)
    log_rets = to_log(prices)

    pf_log_ret = (log_rets * weights).sum(axis=1)
    pf_cumsum = np.exp(pf_log_ret.cumsum()) - 1
    return pf_cumsum


def plot_base_vs_best(base_weights, best_weights, start, end):
    base_curve = build_cumsum_portfolio(base_weights, start, end)
    best_curve = build_cumsum_portfolio(best_weights, start, end)

    plt.figure(figsize=(7, 4))
    plt.plot(base_curve, label="BASE")
    plt.plot(best_curve, label="BEST HEDGE")
    plt.legend()
    plt.title("Cumulative Return: BASE vs BEST HEDGE")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.show()


# ==============================
#             MAIN
# ==============================
def run_engine(portfolio, hedge_universe, max_hedge,
               start="2024-01-01", end="2025-01-01"):

    hedge_combs = generate_hedge_combs(hedge_universe, max_hedge)
    base_pf, weighted_pf = build_base_and_weighted_portfolio(portfolio, max_hedge)
    hedged_pfs = build_hedged_portfolios(base_pf, weighted_pf, hedge_combs)

    evaluate_hedged_portfolios(hedged_pfs, start, end)
    dd_gain, cagr_loss, scores = score_hedges(hedged_pfs)

    best_idx = int(np.argmax(scores))
    best_pf = hedged_pfs[best_idx + 1]

    return {
        "hedged_portfolio": hedged_pfs,
        "dd_gain": dd_gain,
        "cagr_loss": cagr_loss,
        "score": scores,
        "best": {
            "name": best_pf["name"],
            "dd_gain": dd_gain[best_idx],
            "cagr_loss": cagr_loss[best_idx],
            "score": scores[best_idx],
            "weights": best_pf["weights"],
        },
    }


if __name__ == "__main__":
    start_date = "2024-01-01"
    end_date = "2025-01-01"

    result = run_engine(
        PORTFOLIO,
        HEDGE_UNIVERSE,
        MAX_HEDGE,
        start=start_date,
        end=end_date
    )

    pretty_print_result(result)

    base_weights = result["hedged_portfolio"][0]["weights"]
    best_weights = result["best"]["weights"]
    plot_base_vs_best(base_weights, best_weights, start_date, end_date)
