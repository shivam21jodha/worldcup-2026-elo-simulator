# World Cup 2026 — model predictions (published 15 Jul 2026)

Elo + Monte Carlo (200,000 simulations), trained on 49,506 internationals
since 1872. Code and data in this repo; nothing edited after the matches.

## Semifinal, 15 Jul 2026: England vs Argentina
| outcome | probability |
|---|---|
| England win (in 90') | 24.3% |
| Draw after 90' | 29.2% |
| Argentina win (in 90') | 46.5% |
| **England reach the final** | **37.6%** |

## Final, 19 Jul 2026 (Spain qualified)
| champion | probability |
|---|---|
| Spain | 62.3% |
| England | 11.5% |
| Argentina | 26.2% |

## Model honesty check
Backtest on the 101 WC 2026 matches already played:
accuracy 64.4%, Brier 0.4917, log loss 0.8340
(uniform-guess baseline 1.0986).
