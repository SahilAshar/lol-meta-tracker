# Oracle's Elixir — Stat Definitions Reference

Captured 2026-07-19 from [oracleselixir.com](https://oracleselixir.com) (Definitions page).
Reference for interpreting OE site stats and mapping them to CSV columns during analysis.

## Note on use of averages (OE's own caveat)

OE calculates values **per game, then averages across games** — including "share" and
"per minute" stats (damage share, earned gold, wards, CKPM, etc.). Broadcast stats often
use totals instead, which produces minor discrepancies. OE prefers averages to reduce the
impact of game length. Keep this in mind when comparing our computed numbers against the
OE site or broadcast graphics — and be consistent: our scripts should also average
per-game values, not pool totals, when replicating OE-style stats.

## Draft-modeling relevant stats

These matter most for the draft prediction/evaluation direction:

| Stat | Definition | Why it matters for us |
|---|---|---|
| BLND% | Blind-pick rate: % of games picked *before* lane opponent (not always available) | Direct signal of which champions teams commit to without counter-info — a policy-model feature |
| CTR% | Counter-pick rate: % of games picked *after* lane opponent (not always available) | The flip side: champions held for counter-pick = high-flex/low-commit picks |
| B% | % of games champion was banned (not role-specific) | Ban-phase prior |
| P% | % of games champion was picked *in this role* | Role-conditional pick prior |
| P+B% | % of games banned or picked in any role | "Presence" — our headline meta metric |

**Side-selection history (affects draft data):** As of 2026, blue side is no longer
guaranteed to pick first in draft. OE's parser originally assumed blue-picks-first,
which broke champion-select data for early-2026 games. **Fixed and reparsed as of
Jan 18, 2026** (announcement: "Side-Selection Bug," edit confirms updates deployed).
Implications: (1) old warnings about broken 2026 draft data are resolved — spot-checks
remain cheap insurance; (2) for modeling, *pick order must come from the draft columns,
never inferred from side*.

## Full glossary

| Stat | Definition |
|---|---|
| A | Total assists |
| AGT | Average game time/duration, in minutes |
| APG | Assists per game |
| B% | % of games champion was banned (not tied to a specific role) |
| BLND% | Blind-pick rate: % of games picked before lane opponent (not always available) |
| BN% | Baron control rate |
| CCPM | Crowd control dealt to champions per minute |
| CKPM | Average combined kills per minute (team kills + opponent kills) |
| CS%P15 | Average share of team's total CS post-15-minutes |
| CSD10/15/20 | Average creep score difference at 10/15/20 minutes |
| CSPM | Average monsters + minions killed per minute |
| CTR% | Counter-pick rate: % of games picked after lane opponent (not always available) |
| CWPM | Control wards purchased per minute |
| D | Total deaths |
| DMG% | Damage share: average share of team's total damage to champions |
| DMG%P15 / D%P15 | Average share of team's damage to champions post-15-minutes |
| DPG | Deaths per game |
| DPM | Average damage to champions per minute |
| DRG% | Dragon control rate (elemental drakes only if ELD% present) |
| DTH% | Average share of team's deaths |
| EGPM | Average earned gold per minute (excludes starting gold + inherent generation) |
| EGR | Early-Game Rating (OE composite) |
| ELD% | Elder dragon control rate |
| F3T% | First-to-three-towers rate |
| FB% | First Blood rate (players: kill or assist participation) |
| FBN% | First Baron rate |
| FBV% | First Blood Victim rate |
| FD% | First dragon rate |
| FT% | First tower rate |
| GD10/15/20 | Average gold difference at 10/15/20 minutes |
| GOLD% | Gold share: average share of team's total earned gold |
| GP | Games played |
| GPM | Average gold per minute |
| GPR | Gold percent rating (avg share of game's total gold, relative to 50%) |
| GRB% | Void Grub control rate |
| GSPD | Average gold spent percentage difference |
| GXD10/15/20 | Average gold+experience difference at 10/15/20 minutes |
| HLD% | Rift Herald control rate |
| IWC% | Average % of opponent's invisible wards cleared |
| JNG% | Jungle control: average share of game's total jungle CS |
| K | Total kills |
| KD | Kill-to-death ratio |
| KDA | Total kill/death/assist ratio |
| KP | Kill participation |
| KPG | Kills per game |
| KS% | Kill share: % of team's total kills |
| LNE% | Lane control: average share of game's total lane CS |
| LP | Ladder points |
| MLR | Mid/Late Rating (OE composite) |
| OE Rating / OE Rtg | Oracle's Elixir Performance Rating (composite) |
| P% | % of games champion was picked in this role |
| P+B% | % of games champion was banned or picked in any role ("presence") |
| Pos | Position |
| PPG | Turret plates destroyed per game |
| STL / STLPG / StPG | Neutral objectives stolen (total / per game) |
| TDPG | Tower damage dealt per game |
| VSPM | Vision score per minute |
| VWC% | Average % of opponent's visible wards cleared |
| W / L | Total wins / losses |
| W% | Win percentage |
| WC% | Average % of opponent wards cleared |
| WCPM | Average wards cleared per minute |
| WPM | Average wards placed per minute |
| XPD10/15/20 | Average experience difference at 10/15/20 minutes |

---
*Attribution: Oracle's Elixir (Tim Sevenhuysen). Some OE content is provided courtesy
of Leaguepedia under CC-BY-SA 3.0. OE is not endorsed by Riot Games.*
