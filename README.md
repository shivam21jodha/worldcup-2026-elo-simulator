# Can a simple model call the World Cup?

An Elo + Monte Carlo forecast of the 2026 FIFA World Cup, published **before**
the last two matches were played. Predictions are timestamped in
[`outputs/predictions.md`](outputs/predictions.md) and nothing is edited after
the results; the retro section below is filled in after the final.

## Headline prediction (published 15 Jul 2026, before the second semifinal)

| Champion | Probability |
|---|---|
| Spain | **62.3%** |
| Argentina | 26.2% |
| England | 11.5% |

Semifinal (15 Jul 2026): Argentina 62.4% vs England 37.6% to reach the final.

![Championship probabilities](outputs/championship_probs.png)

## Method

- **Data:** all 49,506 men's internationals since 1872
  ([martj42/international_results](https://github.com/martj42/international_results), CC0).
- **Ratings:** Elo with tournament-weighted K (World Cup 60 down to friendlies 20),
  goal-margin multiplier, and +80 home advantage on non-neutral grounds,
  following the World Football Elo Ratings convention.
- **Draw model:** instead of assuming a draw rate, it is estimated empirically —
  draw frequency as a function of absolute Elo gap over all matches since 1993,
  interpolated at prediction time (29% for even matches, falling to 12% at a
  400-point gap).
- **Knockouts:** if the 90 minutes is drawn, the tie goes to extra time and
  penalties, where the favourite keeps a dampened edge
  (0.5 + 0.4 × (Elo expectancy − 0.5)).
- **Simulation:** 200,000 Monte Carlo runs of the remaining bracket
  (seeded, reproducible).

## Is the model honest? Backtest on this World Cup

Before predicting the future I checked the model against the 101 matches of
this tournament already played, using only pre-match ratings:

- **Accuracy 64.4%** (picking the modal outcome of win/draw/win)
- **Log loss 0.834** vs 1.099 for uniform guessing
- Calibration is close to the diagonal — when the model says 70%, it happens
  roughly 70% of the time:

![Calibration](outputs/calibration.png)

## Why Spain?

Spain enter the final with the highest Elo in the world (2299) after beating
Portugal, Belgium and France without conceding more than one goal. The
trajectory chart shows the last four's ratings since 2018:

![Elo trajectories](outputs/elo_trajectories.png)

## Retro (to be completed after the final on 19 Jul 2026)

- Semifinal result vs prediction: _pending_
- Final result vs prediction: _pending_
- What the model missed: _pending_

## Limitations

- Team-level ratings only: no lineup, injury, or fatigue information.
- The draw model conditions only on rating gap, not on knockout incentives.
- Elo reacts slowly to golden generations arriving (see England's 2026 spike).

## Run it

```
pip install pandas numpy matplotlib
python wc_sim.py
```
