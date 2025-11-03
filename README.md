# Outcome Selection with Algorithmic Learners

We define a new procedure to nudge the selection of desirable outcomes in games played by algorithms. We
consider the case where agents use a learning algorithm to play a repeated game. The innovative feature is to
introduce a correlation device: decision makers update the values assigned to each action given the past actions
performance and a payoff irrelevant message. Messages, which can be either public or private, are correlated
among players. The probability distribution over messages is either fixed or time-varying according to some
welfare criterion. We ask the following questions: do algorithms learn desirable correlated equilibria? Does
information improves welfare and fairness when algorithms compete? We give a partial answer to the above
questions based on simulations.

---

## ✨ What’s inside

* **`Q-learning Diff Eq.py`** — Full simulation engine + parameter sweeps + time difference plot to check stability.
* **`Q-Learning Heatmap.py`** — Simulations of games played with the new algorithm and study of the convergence.

Both scripts:

* implement message-conditioned Q-learning with softmax action selection,
* support long horizons and many random seeds,
* save figures (PDF) and caches (PKL) so you can replot without recomputing.

---

## 📦 Requirements

* Python 3.9+ (3.10/3.11 also fine)
* Packages:

  * `numpy`
  * `scipy` 
  * `matplotlib`
  * `tqdm`
  * `pandas` 
  * `pickle` 

Install the core deps:

```bash
pip install numpy matplotlib tqdm
```

---

## 🧠 Model summary

* **Players**: 2; each has 2 actions.
* **Messages**: A pair of messages `(m1, m2)` drawn from a fixed distribution; each player conditions their policy on the message they observe.
* **Policy**: Softmax over message-specific Q-values with inverse temperature `β(t) = β₀ + k·t` (or constant β).
* **Learning**: Exponentially smoothed Q-updates with step `α`.
* **Outputs**: Trajectories of mixed strategies, social welfare, and classification of terminal (last-iterate and averaged) strategies into equilibrium patterns. Batch runs aggregate frequencies → **heatmaps**.

---

## 🚀 Quick start

Run the full reference script:

```bash
python "Q-learning Diff Eq.py"
```

Or the focused heatmap/trajectory runner:

```bash
python "Q-Learning Heatmap.py"
```

By default the scripts will:

1. Sweep parameters (e.g., over `β₀ × k` or `α × β`) and simulate many seeds.
2. Cache results as `.pkl` to avoid recomputation.
3. Write plots as **PDFs** in the working directory.

Common artifacts you’ll see:

* `heatmaps_varying_k_*.pdf`
* `heatmaps_varying_alpha_*.pdf`
* `qlearning_results*.pdf` (time-series)
* `correlated_eq_*.pkl`, `correlated_nash_*.pkl` (caches)

---

## ⚙️ Key parameters

| Name                    | Meaning                                       | Typical values             |
| ----------------------- | --------------------------------------------- | -------------------------- |
| `T`                     | Horizon (iterations)                          | 5,000 – 50,000             |
| `alpha`                 | Q-value smoothing / learning rate             | 0.01 – 0.2                 |
| `beta0`                 | Initial inverse temperature in softmax        | 0 – 5                      |
| `k`                     | Growth rate for `β(t) = β0 + k·t`             | 0 – 1e-3                   |
| `beta`                  | Constant inverse temperature (for α×β sweeps) | 0 – 10                     |
| `message_probabilities` | Distribution over message pairs               | e.g., `[1/3, 1/3, 1/3, 0]` |
| `seeds`                 | Random seeds per grid cell                    | 10 – 200                   |

You can set these near the top/bottom of each script where the experiments are defined.

---

## 📊 Reproducing the main figures

### 1) Heatmap over β₀ × k

Shows the percentage of runs that end in a target equilibrium pattern as you vary the initial temperature and its growth.

```python
# in either script
heatmap_varying_beta0_k(...)
plot_heatmaps(...)  # saves a side-by-side PDF
```

### 2) Heatmap over α × β

Holds β constant (no growth) and varies the learning rate and temperature.

```python
heatmap_varying_alpha_beta(...)
plot_heatmaps(...)
```

### 3) Strategy evolution (time-series)

Plots the probability of action 0 for each player/message across time; versions exist for single or multiple seeds.

```python
q_plot(results, ...)          # multiple trajectories
q_plot_single(result, ...)    # one trajectory
# in the heatmap script:
q_plot_from_results(filtered_results, ...)
```

---

## 🧪 Typical workflows

* **Fast scan** (coarse grid, few seeds) to locate interesting regions.
* **Refine** (denser grid, more seeds) near boundaries where equilibrium frequency changes sharply.
* **Drill-down** by filtering runs that achieve a specific correlated-equilibrium signature and plot only those trajectories.

---

## 🗂️ Project structure (suggested)

```
.
├── Q-learning Diff Eq.py
├── Q-Learning Heatmap.py
├── data/               # (optional) for PKL caches
├── figs/               # PDFs exported here
└── README.md
```

If you want to keep caches/figures tidy, change the output paths in the plotting/saving helpers to point into `data/` and `figs/`.

---

## 🔧 Troubleshooting

* **Nothing happens / file not found**: Remove or update the `os.chdir(...)` line that points to a machine-specific path.
* **Blank plots**: Make sure the sweep functions actually return arrays (e.g., grid sizes > 0) and that your filters don’t exclude all runs.
* **Slow runs**: Reduce `T`, the grid size, or the number of seeds; rely on `.pkl` caches to re-plot without recomputing.

---

## 🧾 License

MIT (recommended). Add a `LICENSE` file or update this section to your preferred license.

---

## 🤝 Contributing

Issues and PRs are welcome:

* Add new game payoffs or message structures
* Plug in alternative temperature schedules
* Add CSV logging and notebooks for analysis

---



Need me to tweak the README tone (more formal, with badges, or with figure thumbnails)? Say the word and I’ll tailor it.
