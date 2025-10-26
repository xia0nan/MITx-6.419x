## CAVIAR Network Analysis – Module 3

Commands to reproduce results and figures for the homework questions. All paths are relative to `coding/module3` and the dataset is read from `coding/module3/CAVIAR/phase*.csv`.

### Prerequisites

- Python 3.10+ in WSL2 (Ubuntu) with your venv activated, or use `uv run` for an ephemeral env.
- Optional for Graphviz layout: `sudo apt-get install graphviz graphviz-dev` and `pip install pygraphviz`.

### Quick start

Run the script (from `coding/module3`):

```bash
python main.py --summary
```

If you prefer `uv` without a venv:

```bash
uv run --with numpy --with pandas --with networkx --with matplotlib python main.py --summary
```

---

### Part (a) – Sizes and visualization

- Print nodes/edges per phase and key phases (2, 6, 10):

```bash
python main.py --summary
```

- Visualize Phase 3 and save an image (with node labels like `n1`, `n3`, …):

```bash
python main.py --describe-phase 3 --plot-phase 3 --with-labels --plot-out phase3_labeled.png
```

Notes:
- If `pygraphviz` is available, a Graphviz layout is used; otherwise the script falls back to a spring layout.

---

### Part (b) – Degree centrality

Normalized degree centrality is `k_i / (n - 1)`. Use:

```bash
# Phase 3
python main.py --describe-phase 3 --centrality n1 n3 n12 n83

# Phase 9
python main.py --describe-phase 9 --centrality n1 n3 n12 n83
```

The script prints values with six decimals (you can round to three significant figures for submission).

---

### Part (b) – Betweenness centrality

Normalized betweenness for undirected graphs (as in the prompt). Use:

```bash
# Phase 3
python main.py --describe-phase 3 --betweenness n1 n3 n12 n83

# Phase 9
python main.py --describe-phase 9 --betweenness n1 n3 n12 n83
```

---

### Part (b) – Eigenvector centrality

Eigenvector centrality with L2-normalization (as in `networkx.eigenvector_centrality`). Use:

```bash
# Phase 3
python main.py --describe-phase 3 --eigenvector n1 n3 n12 n83

# Phase 9
python main.py --describe-phase 9 --eigenvector n1 n3 n12 n83
```

---

### Extra utilities

- Print structural stats (components, degree summary) for any phase:

```bash
python main.py --describe-phase 3 --stats
```

- Plot any other phase (optionally with labels):

```bash
python main.py --plot-phase 9 --with-labels --plot-out phase9_labeled.png
```

---

### Notes

- The script converts each phase’s directed weighted matrix to an undirected binary graph by OR-ing edges in either direction and removing self-loops.
- Only active nodes (degree > 0) are present in the graphs; counts printed with `--summary` also list the total matrix size.

---

## Part (b) – Question 5: Temporal consistency (means across 11 phases)

Compute the mean centrality for each player across all 11 phases, filling 0 for phases in which the player is absent. By default, the script evaluates the 23 players listed in the problem statement (the “Serero organization”).

- Top 3 by mean betweenness centrality (ALL actors). Print IDs only for pasting into the grader:

```bash
python main.py --top-temporal betweenness --only-ids
```

- Top 3 by mean eigenvector centrality (ALL actors). Print IDs only:

```bash
python main.py --top-temporal eigenvector --only-ids
```

If you want to restrict to an explicit list of players, add `--players` (IDs may be given as `n1 n3 ...` or bare `1 3 ...`). Append `--only-ids` to output just the integers:

```bash
python main.py --top-temporal betweenness --players n1 n3 n12 n83 --only-ids

# Restrict to the 23 suspects mentioned in the writeup
python main.py --top-temporal betweenness --suspects-only --only-ids
python main.py --top-temporal eigenvector --suspects-only --only-ids
```


