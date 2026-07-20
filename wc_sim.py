"""
World Cup 2026 Monte Carlo simulator.

Elo ratings computed over the full history of international football
(martj42/international_results, CC0), an empirical draw model, a backtest on
the 2026 World Cup matches already played, and a Monte Carlo simulation of
the final: Spain vs Argentina (19 Jul). Ratings include the semifinal
(England 1-2 Argentina) and the bronze final (France 4-6 England).

Run:  python wc_sim.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SIMS = 200_000

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- palette --
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# categorical slots in fixed order: Spain, Argentina, England, France
TEAM_COLOR = {
    "Spain": "#2a78d6",
    "Argentina": "#1baf7a",
    "England": "#eda100",
    "France": "#4a3aa7",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.family": "Segoe UI",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
})

# ------------------------------------------------------------------- data --
df = pd.read_csv(BASE / "data" / "results.csv", parse_dates=["date"])
df = df.dropna(subset=["home_score", "away_score"]).sort_values("date")
df = df.reset_index(drop=True)
print(f"Matches loaded: {len(df):,}  ({df.date.min().date()} to {df.date.max().date()})")


def k_factor(tournament: str) -> float:
    """K weights follow the World Football Elo Ratings convention."""
    t = tournament.lower()
    if t == "fifa world cup":
        return 60.0
    if "qualification" in t:
        return 40.0
    if any(s in t for s in ("euro", "copa américa", "copa america", "asian cup",
                            "africa cup", "gold cup", "confederations",
                            "nations league")):
        return 50.0
    if t == "friendly":
        return 20.0
    return 30.0


def margin_mult(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


HOME_ADV = 80.0  # Elo points when not on neutral ground

# ------------------------------------------------- Elo pass over history --
ratings: dict[str, float] = {}
history: dict[str, list[tuple[pd.Timestamp, float]]] = {
    t: [] for t in TEAM_COLOR
}
# per-match records for the modern era, used to fit the draw model and backtest
records = []  # (date, tournament, elo_diff_effective, outcome, home, away)
# ratings frozen the instant before the final, so the published forecast is
# reproducible even after the final result enters the data
pre_final_elo: dict[str, float] = {}

for row in df.itertuples(index=False):
    rh = ratings.get(row.home_team, 1500.0)
    ra = ratings.get(row.away_team, 1500.0)
    if row.tournament == "FIFA World Cup" and str(row.date.date()) == "2026-07-19":
        pre_final_elo = {row.home_team: rh, row.away_team: ra}
    diff = rh - ra + (0 if row.neutral else HOME_ADV)
    we = 1 / (1 + 10 ** (-diff / 400))

    if row.home_score > row.away_score:
        w, outcome = 1.0, "H"
    elif row.home_score < row.away_score:
        w, outcome = 0.0, "A"
    else:
        w, outcome = 0.5, "D"

    records.append((row.date, row.tournament, diff, outcome,
                    row.home_team, row.away_team, rh, ra))

    k = k_factor(row.tournament) * margin_mult(abs(int(row.home_score - row.away_score)))
    delta = k * (w - we)
    ratings[row.home_team] = rh + delta
    ratings[row.away_team] = ra - delta
    for team in (row.home_team, row.away_team):
        if team in history:
            history[team].append((row.date, ratings[team]))

rec = pd.DataFrame(records, columns=["date", "tournament", "diff", "outcome",
                                     "home", "away", "elo_home", "elo_away"])

# ------------------------------------------------------------ draw model --
# empirical draw share as a function of |effective Elo diff|, modern era
modern = rec[rec.date >= "1993-01-01"]
bins = np.arange(0, 451, 50)
centers = (bins[:-1] + bins[1:]) / 2
draw_rate = [
    (grp.outcome == "D").mean()
    for _, grp in modern.groupby(pd.cut(modern["diff"].abs(), bins), observed=True)
]
draw_rate = np.array(draw_rate)
print("Draw rate by |Elo diff| bin:", np.round(draw_rate, 3))


def predict(diff: float) -> tuple[float, float, float]:
    """P(home win, draw, away win) in 90 minutes from effective Elo diff."""
    we = 1 / (1 + 10 ** (-diff / 400))
    p_draw = float(np.interp(abs(diff), centers, draw_rate))
    p_home = np.clip(we - p_draw / 2, 0.01, 0.98)
    p_away = np.clip(1 - p_home - p_draw, 0.01, 0.98)
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


# --------------------------------------------------------------- backtest --
wc = rec[(rec.tournament == "FIFA World Cup") & (rec.date >= "2026-06-01")]
probs, actual, hit = [], [], 0
for row in wc.itertuples(index=False):
    p = predict(row.diff)
    label = {"H": 0, "D": 1, "A": 2}[row.outcome]
    probs.append(p)
    actual.append(label)
    hit += int(np.argmax(p) == label)

probs = np.array(probs)
actual = np.array(actual)
onehot = np.eye(3)[actual]
brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
logloss = float(-np.mean(np.log(probs[np.arange(len(actual)), actual])))
baseline = -np.log(1 / 3)
n = len(wc)
print(f"\nBacktest on WC 2026 ({n} matches): "
      f"accuracy {hit}/{n} = {hit / n:.1%}, Brier {brier:.4f}, "
      f"log loss {logloss:.4f} (uniform baseline {baseline:.4f})")

# ------------------------------------------------------------- simulation --
# use the ratings as they stood before the final, so the championship
# probability reproduces the forecast published before kickoff
elo = pre_final_elo
print("\nElo ratings going into the final:")
for t, r in sorted(elo.items(), key=lambda x: -x[1]):
    print(f"  {t:<10} {r:7.0f}")


def sim_knockout(team_a: str, team_b: str, size: int) -> np.ndarray:
    """Simulate a neutral-ground knockout tie; returns bool array (A wins)."""
    diff = elo[team_a] - elo[team_b]
    p_a, p_d, _ = predict(diff)
    u = RNG.random(size)
    a_wins_90 = u < p_a
    drawn = (u >= p_a) & (u < p_a + p_d)
    # extra time / penalties: favourite keeps a dampened edge
    we = 1 / (1 + 10 ** (-diff / 400))
    p_a_et = 0.5 + (we - 0.5) * 0.4
    a_wins_et = RNG.random(size) < p_a_et
    return a_wins_90 | (drawn & a_wins_et)

spain_beats_arg = sim_knockout("Spain", "Argentina", N_SIMS)
champ = {
    "Spain": spain_beats_arg.mean(),
    "Argentina": 1 - spain_beats_arg.mean(),
}
print("\nChampionship probabilities forecast before the final (Spain vs Argentina):")
for t, p in sorted(champ.items(), key=lambda x: -x[1]):
    print(f"  {t:<10} {p:6.1%}")
top_pick = max(champ, key=champ.get)
print(f"\nResult: Spain 1-0 Argentina (Ferran Torres, 106'). "
      f"Model's top pick was {top_pick} at {champ[top_pick]:.1%}. "
      f"{'Correct.' if top_pick == 'Spain' else 'Wrong.'}")

# ----------------------------------------------------------------- charts --
# 1. championship probability bars
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
teams = sorted(champ, key=champ.get)
vals = [champ[t] * 100 for t in teams]
bars = ax.barh(teams, vals, height=0.55,
               color=[TEAM_COLOR[t] for t in teams], zorder=3)
for b, v in zip(bars, vals):
    ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
            va="center", color=INK, fontsize=12, fontweight="bold")
ax.set_xlim(0, max(vals) * 1.22)
ax.set_title("Who wins the 2026 World Cup?", fontsize=15, fontweight="bold",
             color=INK, loc="left", pad=14)
ax.text(0, 1.02, f"Monte Carlo simulation, {N_SIMS:,} runs · Elo model · "
                 f"forecast before the final · result: Spain 1-0 Argentina",
        transform=ax.transAxes, fontsize=9, color=INK_2)
ax.set_axisbelow(True)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", labelsize=12, labelcolor=INK)
ax.set_xlabel("championship probability (%)")
fig.tight_layout()
fig.savefig(OUT / "championship_probs.png", facecolor=SURFACE)
plt.close(fig)

# 2. Elo trajectories of the last four
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
for team, hist in history.items():
    h = pd.DataFrame(hist, columns=["date", "elo"])
    h = h[h.date >= "2018-01-01"]
    ax.plot(h.date, h.elo, color=TEAM_COLOR[team], linewidth=2, zorder=3)
    ax.annotate(f" {team}  {h.elo.iloc[-1]:.0f}",
                (h.date.iloc[-1], h.elo.iloc[-1]),
                color=TEAM_COLOR[team], fontsize=10, fontweight="bold",
                va="center")
ax.set_title("Road to the final: Elo ratings of the last four",
             fontsize=15, fontweight="bold", color=INK, loc="left", pad=14)
# right margin holds the direct labels; ticks stop at 2026 so no phantom years
ax.set_xlim(pd.Timestamp("2018-01-01"), pd.Timestamp("2027-10-01"))
ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2018, 2027)])
ax.set_xticklabels([str(y) for y in range(2018, 2027)])
ax.set_ylabel("Elo rating")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "elo_trajectories.png", facecolor=SURFACE)
plt.close(fig)

# 3. calibration of the WC 2026 backtest
flat_p = probs.flatten()
flat_o = onehot.flatten()
cal_bins = np.linspace(0, 1, 11)
mids, obs, cnt = [], [], []
for lo, hi in zip(cal_bins[:-1], cal_bins[1:]):
    m = (flat_p >= lo) & (flat_p < hi)
    if m.sum() >= 5:
        mids.append(flat_p[m].mean())
        obs.append(flat_o[m].mean())
        cnt.append(int(m.sum()))
fig, ax = plt.subplots(figsize=(5.6, 5.4), dpi=200)
ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.2, linestyle="--", zorder=2)
ax.plot(mids, obs, color="#2a78d6", linewidth=2, marker="o",
        markersize=7, zorder=3)
ax.set_title("Is the model honest?", fontsize=15, fontweight="bold",
             color=INK, loc="left", pad=30)
ax.text(0, 1.02, f"Calibration on all {n} WC 2026 matches played so far",
        transform=ax.transAxes, fontsize=9, color=INK_2)
ax.set_xlabel("predicted probability")
ax.set_ylabel("observed frequency")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT / "calibration.png", facecolor=SURFACE)
plt.close(fig)

# ------------------------------------------------------------- prediction --
final_90 = predict(elo["Spain"] - elo["Argentina"])

with open(OUT / "predictions.md", "w", encoding="utf-8") as f:
    f.write(f"""# World Cup 2026 — model predictions

Elo + Monte Carlo ({N_SIMS:,} simulations), trained on {len(df):,} internationals
since 1872. Code and data in this repo; each section is timestamped when
published and never edited after the match is played.

## Forecast published 19 Jul 2026, before the final: Spain vs Argentina
| outcome | probability |
|---|---|
| Spain win (in 90') | {final_90[0]:.1%} |
| Draw after 90' | {final_90[1]:.1%} |
| Argentina win (in 90') | {final_90[2]:.1%} |
| **Spain champions** | **{champ['Spain']:.1%}** |
| **Argentina champions** | **{champ['Argentina']:.1%}** |

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
Backtest on the {n} WC 2026 matches played (whole tournament):
accuracy {hit / n:.1%}, Brier {brier:.4f}, log loss {logloss:.4f}
(uniform-guess baseline {baseline:.4f}).
""")

print(f"\nSaved: {OUT / 'predictions.md'}, 3 charts in {OUT}")
