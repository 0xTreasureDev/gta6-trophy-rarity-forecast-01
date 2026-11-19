[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xTreasureDev/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0xTreasureDev/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

A forecasting program for the [make.fun trophy rarity bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).
For each GTA VI trophy it predicts the distribution of the share of players
who will have earned it by the end of week one, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of the earn rate in percent.

How it works, in short: the recorded page reads are collapsed into distinct
page states weighted by run length and recency; four candidate trajectories
(persistence, a shrunk log-time trend on the logit scale, a saturating
exponential toward a per-trophy ceiling, and an earners-over-players
decomposition that captures dilution) are blended with weights chosen by
in-case backtests; the spread combines horizon, backtest error, volatility,
counting noise and candidate disagreement.

Run it the way the harness does: the program reads the case file at /input
and writes <id>.out to /output. Python 3.13, standard library only.
