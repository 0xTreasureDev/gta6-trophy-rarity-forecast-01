"""gta6-trophy-rarity-forecast-01 entry: a trajectory forecaster for Exophase trophy earn rates.

Run contract (from the kit): `python3 -P -s /bot/main.py` in python:3.13-slim, standard library only,
no network. The grader runs the program once per case with only that case's file in the input
directory; the program reads every `<id>.in` there (JSON, schema "makefun/gta6-trophies-case/v1")
and writes `<id>.out` to the output directory: {"trophies": [{"id": n, "q": [q05, q25, q50, q75,
q95]}, ...]}, five quantiles of each trophy's earn rate (percent, 0..100) at the case's target, in
the order of the case's `levels`. Input/output directories come from $MAKEFUN_INPUT_DIR and
$MAKEFUN_OUTPUT_DIR when set, else /input and /output. Two runs on the same case must agree byte
for byte: everything below is a deterministic function of the case file.

The model, in one paragraph: an earn rate is a ratio, earners over players tracked, and both sides
of it move. For each trophy the program fits several simple trajectories to the read history on the
logit scale (persistence; a line against log-time; a saturating exponential toward a per-trophy
ceiling) plus one fit that models earners and players tracked separately and divides them again at
the target. The candidates are blended with weights earned in a backtest inside the case itself
(fit on an early part of the history, predict the last reads, pool the errors across trophies, per
tier where a tier has enough trophies). A tier-level trend is blended in for trophies whose own
history is short or flat. The spread around the blended centre combines a horizon floor, the
backtest error scaled to the case's horizon, the trophy's own step-to-step logit volatility, a
counting-noise term from the last read's earners and players (small counts, wide spread), and the
disagreement between the candidate trajectories. One special regime is handled on its own: a
history whose every read is the same page (the launch-night plateau before Exophase's first
refresh) carries no trend at all, so the forecast centres a little below the last rate (a refresh
mostly adds tracked players faster than earners, which dilutes rates) and widens the spread a lot.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; nothing else is imported, but keep the
# line so a sibling module would be found if one is ever shipped.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

RATE_LO = 0.01    # the scoring clamps rates into [0.01, 99.99] before the logit
RATE_HI = 99.99
ANCHOR_FALLBACK = "2026-11-19T00:00:00Z"  # release midnight UTC, the log-time origin


# --------------------------------------------------------------------------- small numeric helpers

def clamp_rate(x):
    return min(RATE_HI, max(RATE_LO, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the house convention); whole numbers come back as ints."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """ISO 8601 UTC text -> epoch seconds (float)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def wls(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x. Returns (a, b) or None when the fit is degenerate."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return (my - b * mx, b)


# --------------------------------------------------------------------------- the case, as states

class Case:
    """A case reduced to its distinct page states.

    Consecutive reads that show the same players-tracked count and the same rate for every trophy
    are one state of the page (Exophase had not refreshed between them); a state is stamped with
    the last read of its run and weighted by the run's length, capped, so a long plateau anchors a
    fit without drowning the reads that actually moved.
    """

    def __init__(self, case):
        self.raw = case
        trophies = case["trophies"]
        self.ids = [t["id"] for t in trophies]
        self.tiers = [t.get("tier", "bronze") for t in trophies]
        self.n = len(trophies)
        game = case.get("game") or {}
        release = str(game.get("release_date") or "2026-11-19")
        try:
            self.anchor = parse_time(release + "T00:00:00Z" if len(release) == 10 else release)
        except ValueError:
            self.anchor = parse_time(ANCHOR_FALLBACK)
        self.target_t = (parse_time(case["target"]) - self.anchor) / 3600.0  # hours from release
        self.horizon_days = case["horizon_minutes"] / 1440.0

        states = []  # (t_hours, players, rates list, run length)
        for read in case["history"]:
            t = (parse_time(read["at"]) - self.anchor) / 3600.0
            players = read.get("players_tracked")
            rates = list(read.get("rates") or [])
            if states and states[-1][1] == players and states[-1][2] == rates:
                prev = states[-1]
                states[-1] = (t, players, rates, prev[3] + 1)
            else:
                states.append((t, players, rates, 1))
        self.states = states
        self.flat = len(states) <= 1

    def hours(self):
        return [s[0] for s in self.states]

    def weights(self):
        return [min(3.0, float(s[3])) * (0.35 + 0.65 * (i / max(1, len(self.states) - 1)))
                for i, s in enumerate(self.states)]

    def series(self, idx, upto=None):
        """(t, logit rate) observations of trophy idx over the first `upto` states."""
        out = []
        states = self.states if upto is None else self.states[:upto]
        for t, _players, rates, _run in states:
            if idx < len(rates) and rates[idx] is not None:
                out.append((t, logit(rates[idx])))
        return out

    def last_rate(self, idx):
        for _t, _p, rates, _r in reversed(self.states):
            if idx < len(rates) and rates[idx] is not None:
                return rates[idx]
        return None

    def last_players(self):
        for _t, players, _rates, _r in reversed(self.states):
            if players:
                return players
        return None


def logtime(t_hours):
    """ln(1 + hours since release), the house curve baseline's clock; negative times sit at 0."""
    return math.log1p(max(0.0, t_hours))


# --------------------------------------------------------------------------- candidate trajectories
# Each candidate maps (case, trophy index, state limit) -> forecast logit at the target, or None
# when it cannot be fit. `upto` restricts the fit to the first `upto` states (the backtest).

def cand_persist(case, idx, upto=None):
    obs = case.series(idx, upto)
    if not obs:
        return None
    return obs[-1][1]


def _move_caps(case):
    d = math.sqrt(max(case.horizon_days, 0.0))
    return 0.7 + 1.05 * d, 0.5 + 0.8 * d  # up, down: how far a centre may travel from the last logit


def _clamp_move(case, idx, value, upto=None):
    obs = case.series(idx, upto)
    if not obs:
        return value
    last = obs[-1][1]
    up, down = _move_caps(case)
    return min(last + up, max(last - down, value))


def cand_linlog(case, idx, upto=None):
    obs = case.series(idx, upto)
    if len(obs) < 3:
        return cand_persist(case, idx, upto)
    states = case.states if upto is None else case.states[:upto]
    ws_all = case.weights()
    xs, ys, ws = [], [], []
    pos = 0
    for j, (_t, _p, rates, _r) in enumerate(states):
        if idx < len(rates) and rates[idx] is not None:
            xs.append(logtime(obs[pos][0]))
            ys.append(obs[pos][1])
            ws.append(ws_all[j])
            pos += 1
    fit = wls(xs, ys, ws)
    if fit is None:
        return cand_persist(case, idx, upto)
    a, b = fit
    # Shrink the slope toward 0 when the trophy's own points are few or barely scattered in time.
    span = (max(xs) - min(xs)) if xs else 0.0
    shrink = span / (span + 1.5) * len(xs) / (len(xs) + 3.0)
    pred = a + b * shrink * logtime(case.target_t) + b * (1.0 - shrink) * xs[-1] if xs else a
    return _clamp_move(case, idx, pred, upto)


def cand_satexp(case, idx, upto=None):
    """Logit rate as a saturating exponential toward a ceiling: L(t) = U + B * exp(-k * t_days)."""
    obs = case.series(idx, upto)
    if len(obs) < 4:
        return cand_persist(case, idx, upto)
    states = case.states if upto is None else case.states[:upto]
    ws_all = case.weights()
    ts, ys, ws = [], [], []
    pos = 0
    for j, (_t, _p, rates, _r) in enumerate(states):
        if idx < len(rates) and rates[idx] is not None:
            ts.append(obs[pos][0] / 24.0)
            ys.append(obs[pos][1])
            ws.append(ws_all[j])
            pos += 1
    t_star = case.target_t / 24.0
    best = None
    for k in (0.1, 0.2, 0.35, 0.6, 1.0, 1.7, 3.0):
        xs = [math.exp(-k * t) for t in ts]
        fit = wls(xs, ys, ws)
        if fit is None:
            continue
        a, b = fit
        sse = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws))
        if best is None or sse < best[0]:
            best = (sse, a, b, k)
    if best is None:
        return cand_persist(case, idx, upto)
    _sse, a, b, k = best
    ceiling = min(a, logit(99.9))
    pred = ceiling + b * math.exp(-k * t_star)
    return _clamp_move(case, idx, pred, upto)


def _forecast_players(case, upto=None):
    """Forecast players tracked at the target: blend of a log-time line on log N and a saturating
    exponential on N itself. Returns None when the history names no players."""
    states = case.states if upto is None else case.states[:upto]
    pts = [(t, p) for (t, p, _r, _run) in states if p]
    if not pts:
        return None
    ws_all = case.weights()
    ws = [ws_all[j] for j, (t, p, _r, _run) in enumerate(states) if p]
    last_n = pts[-1][1]
    preds = []
    fit = wls([logtime(t) for t, _p in pts], [math.log(p) for _t, p in pts], ws)
    if fit is not None:
        a, b = fit
        preds.append(math.exp(a + max(0.0, b) * logtime(case.target_t)))
    best = None
    for k in (0.15, 0.3, 0.6, 1.2, 2.5):
        xs = [math.exp(-k * (t / 24.0)) for t, _p in pts]
        f2 = wls(xs, [float(p) for _t, p in pts], ws)
        if f2 is None:
            continue
        a, b = f2
        sse = sum(w * (p - (a + b * x)) ** 2 for x, (_t, p), w in zip(xs, pts, ws))
        if best is None or sse < best[0]:
            best = (sse, a, b, k)
    if best is not None:
        _sse, a, b, k = best
        preds.append(a + b * math.exp(-k * (case.target_t / 24.0)))
    if not preds:
        return float(last_n)
    n_star = median(preds)
    return min(max(n_star, float(last_n)), float(last_n) * 60.0)


def cand_earners(case, idx, upto=None):
    """Model earners and players separately, then divide: rate = E / N at the target."""
    states = case.states if upto is None else case.states[:upto]
    n_star = _forecast_players(case, upto)
    if not n_star:
        return None
    ws_all = case.weights()
    xs, ys, ws = [], [], []
    last_e = None
    for j, (_t, players, rates, _r) in enumerate(states):
        if players and idx < len(rates) and rates[idx] is not None:
            e = rates[idx] * players / 100.0
            last_e = e
            xs.append(logtime(_t))
            ys.append(math.log(e + 0.35))
            ws.append(ws_all[j])
    if len(xs) < 3 or last_e is None:
        return cand_persist(case, idx, upto)
    fit = wls(xs, ys, ws)
    if fit is None:
        return cand_persist(case, idx, upto)
    a, b = fit
    b = max(-0.02, b)  # earners essentially never shrink; a hair of slack for rounding noise
    e_star = math.exp(a + b * logtime(case.target_t)) - 0.35
    e_star = min(max(e_star, last_e * 0.97), n_star * 0.99995)
    rate = 100.0 * e_star / n_star
    return _clamp_move(case, idx, logit(rate), upto)


CANDIDATES = [
    ("persist", cand_persist),
    ("linlog", cand_linlog),
    ("satexp", cand_satexp),
    ("earners", cand_earners),
]


# --------------------------------------------------------------------------- backtest and blending

def backtest(case):
    """Fit every candidate on early states and score it against the last state, pooled.

    Returns (weights, resid_scale, horizon_bt_days): a weight per candidate name, the ensemble's
    pooled absolute logit error at the backtest horizon, and that horizon in days. With too few
    states there is nothing to backtest and prior weights come back instead.
    """
    prior = {"persist": 0.20, "linlog": 0.30, "satexp": 0.20, "earners": 0.30}
    n = len(case.states)
    if n < 8:
        return prior, None, None
    cuts = sorted({max(4, int(n * f)) for f in (0.5, 0.7, 0.85) if int(n * f) < n})
    errs = {name: [] for name, _fn in CANDIDATES}
    ens_errs = []
    horizons = []
    t_last = case.states[-1][0]
    saved_target = case.target_t
    for cut in cuts:
        # Candidates aim at case.target_t; point them at the last state for this exercise.
        case.target_t = t_last
        horizons.append((t_last - case.states[cut - 1][0]) / 24.0)
        preds = {}
        for name, fn in CANDIDATES:
            got = []
            for idx in range(case.n):
                obs = case.series(idx, cut)
                if not obs:
                    continue
                actual = case.series(idx)[-1][1] if case.series(idx) else None
                if actual is None:
                    continue
                pred = fn(case, idx, cut)
                if pred is not None:
                    got.append((idx, pred, actual))
            preds[name] = got
            for _idx, pred, actual in got:
                errs[name].append(abs(pred - actual))
        # The ensemble with prior weights, for the residual scale.
        by_idx = {}
        for name, got in preds.items():
            for idx, pred, _actual in got:
                by_idx.setdefault(idx, []).append((prior[name], pred))
        for idx, parts in by_idx.items():
            actual = case.series(idx)[-1][1]
            centre = sum(w * p for w, p in parts) / sum(w for w, _p in parts)
            ens_errs.append(abs(centre - actual))
    case.target_t = saved_target
    weights = {}
    for name, _fn in CANDIDATES:
        pool = errs[name]
        mae = sum(pool) / len(pool) if pool else 1.0
        weights[name] = 1.0 / (mae + 0.05) ** 2
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    resid = 1.253 * (sum(ens_errs) / len(ens_errs)) if ens_errs else None
    horizon_bt = median(horizons) if horizons else None
    return weights, resid, horizon_bt


def tier_trends(case):
    """Per-tier median log-time slope of the logit rate, from trophies with a fittable series."""
    by_tier = {}
    for idx in range(case.n):
        obs = case.series(idx)
        if len(obs) < 3:
            continue
        fit = wls([logtime(t) for t, _l in obs], [l for _t, l in obs],
                  [1.0] * len(obs))
        if fit is not None:
            by_tier.setdefault(case.tiers[idx], []).append(fit[1])
    return {tier: median(slopes) for tier, slopes in by_tier.items() if slopes}


def step_volatility(case, idx):
    """The trophy's own logit step volatility per sqrt(day), from consecutive states."""
    obs = case.series(idx)
    num = 0.0
    den = 0.0
    for (t0, l0), (t1, l1) in zip(obs, obs[1:]):
        dt = (t1 - t0) / 24.0
        if dt <= 0:
            continue
        dl = max(-2.0, min(2.0, l1 - l0))
        num += dl * dl
        den += dt
    if den <= 0:
        return 0.0
    return math.sqrt(num / den)


# --------------------------------------------------------------------------- the forecast itself

def forecast_case(case_dict):
    case = Case(case_dict)
    if not case.states:
        raise ValueError("case has no history")
    weights, resid_scale, horizon_bt = backtest(case)
    trends = tier_trends(case)
    t_last = case.states[-1][0]
    du = logtime(case.target_t) - logtime(t_last)
    players_last = case.last_players()
    horizon_floor = math.sqrt((0.35 * math.sqrt(case.horizon_days)) ** 2 + 0.12 ** 2)
    if resid_scale is not None and horizon_bt:
        bt_scale = math.sqrt((case.horizon_days + 0.25) / (horizon_bt + 0.25))
        bt_scale = min(2.2, max(0.8, bt_scale))
    else:
        bt_scale = 1.0

    rows = []
    for idx in range(case.n):
        last = case.last_rate(idx)
        if last is None:
            # No read ever listed this trophy: fall back to a low, wide forecast.
            centre, sigma = logit(1.0), 1.4
            rows.append((case.ids[idx], centre, sigma))
            continue
        l_last = logit(last)
        preds = {}
        for name, fn in CANDIDATES:
            pred = fn(case, idx)
            if pred is not None:
                preds[name] = pred
        if preds:
            wsum = sum(weights[name] for name in preds)
            centre = sum(weights[name] * p for name, p in preds.items()) / wsum
            disagree = math.sqrt(
                sum(weights[name] * (p - centre) ** 2 for name, p in preds.items()) / wsum)
        else:
            centre, disagree = l_last, 0.0

        # Borrow the tier's trend when this trophy's own series is short or has barely moved.
        distinct = len({round(l, 6) for _t, l in case.series(idx)})
        w_group = 0.45 if distinct <= 2 else (0.25 if distinct <= 4 else 0.12)
        slope = trends.get(case.tiers[idx])
        if slope is not None and du > 0:
            group_pred = l_last + slope * du
            up, down = _move_caps(case)
            group_pred = min(l_last + up, max(l_last - down, group_pred))
            centre = (1.0 - w_group) * centre + w_group * group_pred

        # Spread: horizon floor, backtest error at this horizon, own step volatility,
        # counting noise from the last read's counts, and candidate disagreement.
        earners_last = last * players_last / 100.0 if players_last else 0.0
        if players_last:
            sigma_cnt = math.sqrt(1.0 / max(earners_last, 0.7)
                                  + 1.0 / max(players_last - earners_last, 0.7))
            sigma_cnt = min(sigma_cnt, 1.2)
        else:
            sigma_cnt = 0.6
        vol = step_volatility(case, idx)
        sigma_rw = vol * math.sqrt(case.horizon_days)
        sigma_bt = (resid_scale or 0.0) * bt_scale
        sigma = math.sqrt(horizon_floor ** 2 + (0.8 * sigma_bt) ** 2 + (0.7 * sigma_rw) ** 2
                          + sigma_cnt ** 2 + (0.7 * disagree) ** 2)
        # The terms overlap (floor, backtest error and step volatility all price the same
        # wander), so the quadrature sum is calibrated down by a fixed factor, chosen on
        # simulated launch weeks: 1.0 priced every band too wide, 0.5 began to drop coverage
        # when the world's behaviour shifted after the checkpoint, 0.72 kept both.
        sigma = min(2.4, max(0.18, sigma * 0.72))

        if case.flat:
            # The plateau regime: no refresh anywhere in the history. A refresh before the
            # target mostly grows the tracked base faster than earners, diluting rates, and
            # its size is the dominant unknown: centre a shade lower, spread much wider.
            span_h = case.states[-1][0] - case.states[0][0] if len(case.states) > 1 else 0.0
            shift_scale = 1.0 if span_h <= 12.0 else 0.4
            centre = l_last - 0.85 * shift_scale * (1.0 - clamp_rate(last) / 100.0)
            if last >= RATE_HI:
                centre = logit(99.9)
            sigma = max(sigma, 1.05)
        rows.append((case.ids[idx], centre, sigma))

    out_rows = []
    for tid, centre, sigma in rows:
        centre = min(logit(99.95), max(logit(0.01), centre))
        q = []
        for z in Z:
            v = round4(100.0 * sigmoid(centre + z * sigma))
            v = min(100, max(0, v))          # sigmoid keeps this in range; a guard, not a clamp
            if q and v < q[-1]:              # rounding must never break the non-decreasing rule
                v = q[-1]
            q.append(v)
        out_rows.append({"id": tid, "q": q})
    return {"trophies": out_rows}


def main(argv):
    input_dir = INPUT_DIR
    paths = []
    if len(argv) > 1:
        given = Path(argv[1])
        if given.is_dir():
            input_dir = given
            paths = sorted(input_dir.glob("*.in"))
        elif given.is_file():
            paths = [given]
    else:
        paths = sorted(input_dir.glob("*.in"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") not in (None, LEVELS):
            raise ValueError(f"{path.name} asks for levels {case.get('levels')}, not {LEVELS}")
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
