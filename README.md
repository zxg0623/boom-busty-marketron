# Reproducing the Boom-Busty Marketron

English · [简体中文](README.zh-CN.md)

This repository reconstructs the two market models in Igor Halperin’s *Boom-Busty Marketron as a Model of Market Dynamics*. Both ask what happens when investors respond to prices and their demand, in turn, moves prices. In the 500-stock reconstruction, switching off that feedback reduces estimated entropy production by about 80%. Bear markets still occur in both settings, although the baseline generates about half as many as the paper’s model. [Paper, version 1](https://arxiv.org/abs/2611.01185v1) · [Full text, equations and tables](https://arxiv.org/html/2611.01185v1)

## How the models work

**Multi-asset BBM.** Each stock’s log price has three components, $\log S = \log F + x + \eta$. Fundamental value $F$ responds to market-wide and firm-specific shocks. Mispricing $x$ tends to revert toward fair value, and $\eta$ captures short-lived price disturbances. A collective investor uses recent changes in mispricing, remembered demand and perceived risk to choose target holdings. Cash, position limits and transaction costs constrain the trades it can execute. The investor’s demand memory feeds back into mispricing through the impact coefficient $c$. Setting $c=0$ switches off that channel. [Multi-asset simulator](model.py)

The full simulation keeps separate books for the ten largest stocks and the other 490. Both groups experience the same market shocks. Each day’s index return uses the previous close’s market capitalizations. When a firm leaves the index, its successor enters after that day’s return is calculated. The model continues to follow removed firms for up to five years to count subsequent defaults. [Inputs and parameter values](inputs.py)

**Index-only I-BBM.** A separate calibration treats the entire index as one tradable asset. The asset has its own mispricing, demand memory and stochastic volatility. A negative correlation between price and volatility shocks allows a price decline to raise subsequent volatility. This version can reproduce index-level behavior without simulating individual firms or changes in membership. [Index simulator](index_model.py)

## Run the reproduction

The code requires Python 3.8 or later. The archived run used 64-bit Python 3.13.0 on Windows. `requirements.txt` installs the archived pins on Python 3.12+ and the last compatible releases on Python 3.8–3.11; the archived results were produced with the former. Clone the repository and install the dependencies in a virtual environment. In PowerShell, run the following commands. [Dependencies](requirements.txt) · [Archived environment and seed](results/run_info.json)

```powershell
git clone https://github.com/zxg0623/boom-busty-marketron.git
cd boom-busty-marketron
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python verify.py
.\.venv\Scripts\python run.py all --profile quick --jobs 2
```

`verify.py` checks the nonlinear impact step against an independent ODE solver, then checks the portfolio budget and reconstructs index returns from constituent weights. A successful run prints `"passed": true`. The quick profile runs a 100-stock example with three years of BBM warm-up and writes its CSVs and figures to `quick_results/`. Numba compiles the simulation kernels on first use.

For a POSIX shell, create the environment with `python3 -m venv .venv` (Python 3.8 or later) and use `.venv/bin/python` in place of `.\.venv\Scripts\python` in the commands above and below.

To reproduce the results below, run the full profile. It uses 500 stocks, a 50-year BBM warm-up, a five-year I-BBM warm-up and master seed `20261126`. The commands below write into a separate directory so you can compare the output with the archived files.

```powershell
.\.venv\Scripts\python run.py all --profile full --jobs 5 --out my_results
.\.venv\Scripts\python verify.py --out my_results/verification.json
```

After installing dependencies, both profiles run offline. Use `--jobs 1` to run sequentially. To repeat the optional two-parameter fit, run `.\.venv\Scripts\python fit_scales.py --jobs 4 --out my_results`. The separate historical cross-check can retrieve the S&P 500 price and total-return series from Yahoo Finance or read local copies. [Historical cross-check script](historical_check.py)

## Comparison with the paper

Window A uses inputs for **1996–2010** and spans 3,777 daily returns. Window B uses inputs for **2011–2024** and spans 3,522. The paper column below gives the author’s **simulated** results. Historical market estimates are identified separately in the [full comparison table](results/comparison.csv).

| Measure | This reconstruction | Paper model | Source |
|---|---:|---:|---|
| Impact-only absorption time from $x=-1.6,-2,-3$, in trading days | 47.22, 20.86, 4.71 | ≈48, 21, 5 | §3.8 |
| I-BBM annualized index volatility, A / B | 18.49% / 18.49% | 18.3% / 18.4% | Table 11 |
| I-BBM annualized mean log return, A / B | 12.27% / 12.76% | 9.4% / 9.3% | Table 11 |
| BBM annualized index volatility, A / B | 16.09% / 14.92% | 20.4% / 20.8% | Table 5 |
| BBM declines of at least 20% per century, baseline → feedback off | 13.50 → 14.92 | 25.8 → 26.7 | Table 33 |
| BBM entropy production, linear OU estimate, baseline → feedback off (nats/year) | 78.38 → 15.71 | 71.2 → 12.1 | Table 34 |
| I-BBM entropy production, conditional / linear OU estimate (nats/year) | 9.21 / 4.12 | 8.50 / 2.43 | Table 28 |

**Volatility and drawdowns.** The single-index model comes close to the paper’s volatility in both windows but has higher average returns. The 500-stock index has lower volatility and fewer deep declines. An episode starts when the index falls 20% from a local peak. A 20% rebound from the subsequent trough resets the search for a new peak. Applying this rule to the public S&P 500 data gives four episodes in A and two in B, matching the historical frequencies in Table 5. [Episode definitions](diagnostics.py) · [Historical cross-check results](data/historical_crosscheck.csv)

**Feedback and time irreversibility.** The linear Ornstein–Uhlenbeck (OU) estimate measures how strongly the joint paths of mispricing, demand memory and log variance distinguish forward from reversed time. The BBM comparison uses eight paired seeds with window A inputs. Table 33’s event counts use the first 150 years of each path. Table 34’s entropy estimates use all 200 years and include days when constituents change. The paper uses six paths for Table 34. The I-BBM estimates in Table 28 use 32 paths of 200 years each. [Paired differences and standard errors](results/paired_effects.csv)

![Comparison of bear-market frequency and entropy production in the baseline and feedback-off experiments](results/feedback.png)

The short-window comparisons summarize 16 BBM paths per window by the median and the interquartile range divided by 1.349. I-BBM uses the mean and sample standard deviation of 20 paths per window. The reconstruction’s bands overlap the paper’s model bands on 17 of 21 BBM measures in A and 15 of 21 in B. For I-BBM the count is 13 of 14 in each window. The CSV records the center and width on both sides so readers can inspect the size of each discrepancy. [Per-metric comparisons](results/comparison.csv) · [Summary statistics](results/summary.csv)

## Reconstructed inputs and sensitivity

The public v1 source archive includes the article, figures and parameter tables. The original simulator, seeds and point-in-time CRSP stock panel were unavailable for this reconstruction. The BBM baseline takes the joint parameters in Table 4 directly from the author’s two-stage calibration and generates firms from the published size, sector and entry-price statistics. The two group volatility scales also draw on published single-stock volatilities, which limits their value as independent checks. [Input reconstruction notes (Chinese)](data/inputs_zh.md)

One sensitivity test estimates just those two scales from window A’s stock-basket volatilities. On eight separate evaluation seeds for window B, the median annualized index volatility moves from **14.44%** with baseline scales to **24.04%** with fitted scales. The **historical market** reference is **17.2%**. These eight seeds are separate from the 16 used for the main BBM table. The result shows how much the missing stock-level inputs can move the index. [Fitting code](fit_scales.py) · [Sensitivity results](results/scale_fit.csv)

For I-BBM, the maximum of the bare potential in Figure 5 and the saddle in §7.4 supply two parameters omitted from Table 10. The reconstruction uses the common fundamental log-growth rate of 0.1262 from Table 3, which directly affects the drift comparison. The numerical checks also cover finite-time absorption and the potential barrier. [Index model assumptions in code](index_model.py) · [Numerical verification](results/verification.json)

The archived daily paths, drawdown dates, figures and per-seed metrics are available in the [results directory](results/). For a worked example, the first preset seed for window A has both a [daily 500-stock index path](results/BBM_A_path.csv) and a [chart of its index level, drawdown and mispricing](results/bbm_path.png).
