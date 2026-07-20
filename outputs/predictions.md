# World Cup 2026 — model predictions

Elo + Monte Carlo (200,000 simulations), trained on 49,509 internationals
since 1872. Code and data in this repo; each section is timestamped when
published and never edited after the match is played.

## Forecast published 19 Jul 2026, before the final: Spain vs Argentina
| outcome | probability |
|---|---|
| Spain win (in 90') | 39.2% |
| Draw after 90' | 29.0% |
| Argentina win (in 90') | 31.8% |
| **Spain champions** | **54.2%** |
| **Argentina champions** | **45.8%** |

Ratings include the semifinal (England 1-2 Argentina) and the bronze final
(France 4-6 England).

**Result: Spain 1-0 Argentina** (Ferran Torres, 106', extra time). The model's
top pick won. Across the knockouts it called both the semifinal and the final
correctly.

## Track record (published 15 Jul 2026, before the second semifinal)
- Semifinal, England vs Argentina: model gave Argentina 62.4% to reach the
  final. Result: England 1-2 Argentina. Correct call.
- Championship probabilities as of 15 Jul: Spain 62.3%, Argentina 26.2%,
  England 11.5%.

## Model honesty check
Backtest on the 104 WC 2026 matches played (whole tournament):
accuracy 64.4%, Brier 0.4943, log loss 0.8381
(uniform-guess baseline 1.0986).
