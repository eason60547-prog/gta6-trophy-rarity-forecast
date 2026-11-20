"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on Exophase at a case's target time: five quantiles (5/25/50/75/95%) per
trophy. Python 3.13, standard library only, no network, deterministic.

Why not the house baseline
---------------------------
The house baseline (starter/) says every rate stays where the last read put
it, with a random-walk spread. On launch night that scores a pinball loss of
about 1.51 on the practice cases: Exophase refreshes its counts in jumps, and
the refresh that landed at 04:30 UTC on 19 Nov moved every rate outside the
baseline's narrow band (players tracked 638 -> 3,868, every rate but the one
at 100% lower). Persistence is the wrong model when the denominator jumps.

The model here: earners and players separately
----------------------------------------------
A trophy's rate is earners / players_tracked. The two move for different
reasons, so they are forecast separately and divided at the end:

* Players tracked N(t). Exophase's page is a stale snapshot: it only changes
  when Exophase refreshes, so the last read's N is as of the last refresh, not
  the checkpoint. N(t) is fitted from the case's distinct snapshots (deduped
  reads) as a power law in hours since launch, log N = a + b*log(t), against
  an exponential alternative, keeping the better in-sample fit; the target N
  is that fit read at the target, capped at 8x the last read. With a single
  frozen snapshot (the practice cases: 7-8 identical reads), there is nothing
  to fit, so a launch-night prior is used instead: on launch night N doubled
  roughly every 1.8 hours (638 players at 23:50 UTC on 18 Nov, 3,868 at the
  04:30 refresh on 19 Nov, per the bounty rules).

* Earners E_i(t). For each trophy, earners are computed at every distinct
  snapshot (rate * N / 100) and fitted against cumulative player-hours
  X(t) = integral of N dt, in log-log space: log E_i = a_i + b_i * log X.
  The slope is shrunk toward its tier's mean slope (platinum/gold/silver/
  bronze) and clipped to [0, 1.2]. X at the target comes from integrating the
  fitted N(t). Earners are capped so no rate can exceed its tier cap
  (platinum 2%, gold 10%, silver 30%, bronze 60%) or 99.9%.

* The one trophy at 100% ("Complete Gone Fishing", every tracked player has
  it) is forecast at 100% throughout: it is the game's opening beat, and it
  was still 100.00% with ~15,000 players tracked 43 hours after launch.

* Single-snapshot cases (practice): earners are grown from the last read by
  a per-class factor over the horizon (fast early trophies x2.5, medium x1.5,
  slow x1.1), the classes calibrated on the live Exophase page of 20 Nov
  18:29 UTC (43 hours after launch: Chapter 1 at 11.98%, the fishing
  challenge at 13.30%, most grind/collectible trophies still at 0.01%).

* Spread. Quantiles are logit-normal around the median: sigma comes from the
  earners fit's residual spread where there is one (floored 0.25), plus
  0.35 * sqrt(horizon in days), plus 0.15 when the horizon tops a day
  (refresh-timing uncertainty); single-snapshot cases get sigma 0.9. Sigma is
  kept in [0.3, 1.4]. The pinball loss is a proper scoring rule, so the spread
  is set to what is honestly believed, not tuned to look confident.

What is baked in, and what is fitted at run time
-------------------------------------------------
Everything the program knows at run time is in the case file: the launch
night doubling time (1.8 h), the tier caps, and the fast/medium/slow classes
are constants in this file, calibrated on public data read while building
(the bounty rules' own account of the 04:30 refresh, and the live Exophase
page of 20 Nov 2026). Every level and slope that can be fitted is fitted
from the case's own history at run time; nothing is fetched.

Run it
-------
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

The grader runs `python3 -P -s /bot/main.py` with /input and /output. One
output file per input case, <id>.out, {"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}]}, quantiles rounded to 4 decimals.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch hour, inferred: the first house read (23:50 UTC, 18 Nov 2026) already
# showed 638 players tracked, minutes after the game unlocked in Europe.
LAUNCH = datetime(2026, 11, 18, 23, 0, 0, tzinfo=timezone.utc)

# Launch-night prior, from the bounty rules: 638 players at the 23:50 read,
# 3,868 at the 04:30 refresh, 4.67 h later: a doubling time of ~1.8 h.
# Used only when a case's history holds a single distinct snapshot.
LAUNCH_DOUBLING_HOURS = 1.8
# Cap on the players-tracked extrapolation, as a multiple of the last read.
MAX_N_GROWTH = 8.0

# Earners caps by tier (percent): a trophy's forecast rate never exceeds these.
TIER_CAP = {"platinum": 2.0, "gold": 10.0, "silver": 30.0, "bronze": 60.0}
# Fallback earners-growth slope (log earners vs log player-hours) by tier,
# used when a trophy has too few distinct snapshots to fit its own.
TIER_SLOPE_PRIOR = {"platinum": 0.6, "gold": 0.8, "silver": 0.9, "bronze": 1.0}

# Speed classes for the single-snapshot path, calibrated on the live Exophase
# page of 20 Nov 2026 18:29 UTC (43 h after launch). Fast: trophies already
# at 1%+ then (early chapters, the fishing challenge, the breakfast run, the
# arcade, discovering vehicles, bowling). Medium: 0.05-1% then. Slow: 0.01%.
# The multiplier is the earners growth applied over the horizon.
FAST_IDS = {3, 4, 8, 20, 21, 33, 35, 37, 44, 46, 50}
MEDIUM_IDS = {36, 38, 34, 39, 42, 51, 53}
SINGLE_GROWTH = {"fast": 2.5, "medium": 1.5, "slow": 1.1}
SINGLE_SIGMA = 0.9


def parse_time(s):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(dt):
    """Hours from launch, floored so log-time fits never see zero."""
    return max(0.25, (dt - LAUNCH).total_seconds() / 3600.0)


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    """Round to 4 decimals, half up (the house baseline's convention)."""
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def ols(xs, ys):
    """Least squares y = a + b*x; returns (intercept, slope, rmse)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    rmse = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, rmse


def distinct_snapshots(history):
    """Collapse consecutive identical reads into distinct snapshots.

    Exophase's page is frozen between its own refreshes, so most reads in a
    case repeat the previous one exactly. Only changes carry information.
    """
    snaps = []
    for r in history:
        rates = r["rates"]
        n = r["players_tracked"]
        if snaps and snaps[-1]["n"] == n and snaps[-1]["rates"] == rates:
            continue
        snaps.append(
            {
                "t": hours_since_launch(parse_time(r["at"])),
                "at": r["at"],
                "n": n,
                "rates": list(rates),
            }
        )
    return snaps


def fit_players(snaps, t_at, t_target):
    """Forecast players tracked at the target.

    Returns (n_target, n_at, n_fn): the fitted N at the target and at the
    checkpoint, and N as a function of hours since launch. With 2+ distinct
    snapshots, fits log N against log hours (power law) and against hours
    (exponential) and keeps the better fit; both slopes are clamped to sane
    growth (a negative fit means the refreshes disagree, and is read as
    flat). With one snapshot, uses the launch-night doubling prior from the
    time of the first read.
    """
    n_last = snaps[-1]["n"]
    if len(snaps) >= 2:
        xs_log = [math.log(s["t"]) for s in snaps]
        xs_lin = [s["t"] for s in snaps]
        ys = [math.log(s["n"]) for s in snaps]
        a_pow, b_pow, rmse_pow = ols(xs_log, ys)
        a_exp, b_exp, rmse_exp = ols(xs_lin, ys)
        b_pow = min(max(b_pow, 0.0), 2.0)
        b_exp = min(max(b_exp, 0.0), 0.5)
        if rmse_pow <= rmse_exp:
            n_fn = lambda t: math.exp(a_pow + b_pow * math.log(max(t, 0.25)))
        else:
            n_fn = lambda t: math.exp(a_exp + b_exp * t)
        n_target = min(max(n_fn(t_target), n_last), n_last * MAX_N_GROWTH)
        n_at = min(max(n_fn(t_at), n_last), n_last * MAX_N_GROWTH)
        return n_target, n_at, n_fn
    t_first = snaps[0]["t"]
    n0 = snaps[0]["n"]
    n_fn = lambda t: min(
        n0 * 2.0 ** ((t - t_first) / LAUNCH_DOUBLING_HOURS),
        n0 * MAX_N_GROWTH,
    )
    n_target = min(n_fn(t_target), n_last * MAX_N_GROWTH)
    n_at = min(n_fn(t_at), n_last * MAX_N_GROWTH)
    return n_target, n_at, n_fn


def player_hours(snaps, n_fit_fn, t_from, t_to, steps=400):
    """Integral of N(t) from t_from to t_to (hours), trapezoid rule.

    n_fit_fn(t) gives players tracked at hour t. Deterministic: fixed steps.
    """
    if t_to <= t_from:
        return 0.0
    dt = (t_to - t_from) / steps
    total = 0.0
    prev = n_fit_fn(t_from)
    for i in range(1, steps + 1):
        cur = n_fit_fn(t_from + dt * i)
        total += 0.5 * (prev + cur) * dt
        prev = cur
    return total


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    t_at = hours_since_launch(parse_time(case["at"]))
    t_target = hours_since_launch(parse_time(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0

    snaps = distinct_snapshots(history)
    n_target, n_at, n_fn = fit_players(snaps, t_at, t_target)

    # Cumulative player-hours at each distinct snapshot, and at the target.
    exposures = []
    cum = 0.0
    prev_t = snaps[0]["t"]
    prev_n = snaps[0]["n"]
    exposures.append(0.0)
    for s in snaps[1:]:
        cum += 0.5 * (prev_n + s["n"]) * (s["t"] - prev_t)
        exposures.append(cum)
        prev_t, prev_n = s["t"], s["n"]
    # Exposure from the last snapshot to the target, by the fitted N(t).
    x_target = exposures[-1] + player_hours(
        snaps, n_fn, snaps[-1]["t"], t_target
    )

    last_rates = snaps[-1]["rates"]
    n_last = snaps[-1]["n"]

    # Per-trophy earners fits (log earners vs log exposure), with tier
    # shrinkage. Trophies with <3 usable points fall back to their tier prior.
    slopes = {}
    intercepts = {}
    residuals = {}
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        tier_slopes[tier] = []
    for i, tr in enumerate(trophies):
        xs, ys = [], []
        for j, s in enumerate(snaps):
            r = s["rates"][i]
            if r is None or r <= 0:
                continue
            e = r * s["n"] / 100.0
            if e <= 0 or exposures[j] <= 0:
                continue
            xs.append(math.log(exposures[j]))
            ys.append(math.log(e))
        if len(xs) >= 3 and max(xs) - min(xs) > 1e-9:
            a, b, rmse = ols(xs, ys)
            slopes[i] = b
            intercepts[i] = a
            residuals[i] = rmse
            tier_slopes[tr["tier"]].append(b)
    tier_mean = {}
    for tier, vals in tier_slopes.items():
        tier_mean[tier] = sum(vals) / len(vals) if vals else TIER_SLOPE_PRIOR[tier]

    rows = []
    for i, tr in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = 0.01
        # The 100% trophy stays 100%: every tracked player has it, and it
        # was still 100.00% with ~15,000 players tracked 43 h after launch.
        if r_last >= 99.99:
            rows.append({"id": tr["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue
        e_last = max(r_last * n_last / 100.0, 0.01)
        if len(snaps) >= 2 and i in slopes:
            b = slopes[i]
            b = 0.5 * b + 0.5 * tier_mean[tr["tier"]]
            b = min(max(b, 0.0), 1.2)
            log_e = intercepts[i] + b * math.log(max(x_target, 1e-9))
            e_target = math.exp(log_e)
            sig_fit = min(max(residuals[i], 0.25), 0.8)
        elif len(snaps) >= 2:
            b = tier_mean[tr["tier"]]
            e_target = e_last * (max(x_target, 1e-9) / max(exposures[-1], 1e-9)) ** b
            sig_fit = 0.5
        else:
            # Single frozen snapshot: earners grow by class over the horizon.
            if tr["id"] in FAST_IDS:
                g = SINGLE_GROWTH["fast"]
            elif tr["id"] in MEDIUM_IDS:
                g = SINGLE_GROWTH["medium"]
            else:
                g = SINGLE_GROWTH["slow"]
            e_target = e_last * g
            sig_fit = SINGLE_SIGMA
        cap = min(TIER_CAP.get(tr["tier"], 60.0), 99.9)
        e_target = min(e_target, n_target * cap / 100.0)
        median = clamp(100.0 * e_target / n_target)
        sigma = sig_fit + 0.35 * math.sqrt(horizon_days)
        if horizon_days > 1.0:
            sigma += 0.15
        sigma = min(max(sigma, 0.3), 1.4)
        med_logit = logit(median)
        qs = []
        prev = -1.0
        for z in Z:
            q = round4(inv_logit(med_logit + z * sigma))
            q = min(max(q, 0.01), 99.99)
            q = max(q, prev)  # keep the row non-decreasing after rounding
            qs.append(q)
            prev = q
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in INPUT_DIR.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        if case["levels"] != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case['levels']}")
        out = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
