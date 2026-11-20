# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/eason60547-prog/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/eason60547-prog/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one — an entry for make.fun's `gta6-trophy-rarity-forecast-01` bounty ($250 in USDC).

## What it does

`main.py` (Python 3.13, standard library only, no network) reads each case file at `/input/<id>.in` — the house's recording of Exophase's GTA VI PS5 trophy page before the case's checkpoint — and writes `/output/<id>.out`: five quantiles (5/25/50/75/95%) of every trophy's earn rate at the case's target.

## The model

A trophy's rate is earners / players tracked, and the two move for different reasons, so they are forecast separately and divided at the end:

- **Players tracked N(t).** Exophase's page is a stale snapshot — it only changes when Exophase refreshes — so the last read's N is as of the last refresh, not the checkpoint. N(t) is fitted from the case's distinct snapshots as a power law in hours since launch, against an exponential alternative, keeping the better in-sample fit. With a single frozen snapshot (the practice cases), a launch-night prior is used instead: N doubled roughly every 1.8 hours (638 players at 23:50 UTC on 18 Nov, 3,868 at the 04:30 refresh on 19 Nov).
- **Earners E_i(t).** Per trophy, earners at each distinct snapshot are fitted against cumulative player-hours in log-log space, with the slope shrunk toward its tier's mean. The one trophy at 100% stays at 100%.
- **Spread.** Quantiles are logit-normal around the median: sigma from the earners fit's residuals, plus 0.35·√(horizon in days), plus 0.15 past a day for refresh-timing uncertainty. The pinball loss is a proper scoring rule, so the spread is set to what is honestly believed.

The house baseline (persistence + random-walk spread) scores about 1.51 on the practice cases; the refresh that landed at 04:30 on launch night moved every rate outside its band. This entry is built to survive exactly that kind of jump.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=mz84a1xs&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
