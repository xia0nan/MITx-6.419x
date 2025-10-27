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

- Plot two consecutive phases side‑by‑side (helps for Part (f) Q1):

```bash
python main.py --plot-pair 4 --with-labels   # plots phases 4 and 5 into pair_4_5.png
```

- Export all consecutive pairs into `coding/module3/pairs/`:

```bash
python main.py --plot-pairs-all --with-labels
```

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


---

### Part (j) – Hubs and Authorities (HITS) on directed graphs

Use the directed graph for each phase and compute HITS scores:

```bash
# Top hubs and authorities for a given phase (e.g., 3)
python main.py --hits-phase 3 --hits-top 10

# Track specific nodes (e.g., n1 and n3) across all phases and print CSV
python main.py --hits-track n1 n3 > hits_track.csv
```

---

## Parts (f)–(j) – Workflow and Answers

### Part (f) Q1/Q2

- Identify X using the pair plots (`pairs/pair_X_X+1.png`).
- For Q2, compare centralities across phases X and X+1 for key actors:

```bash
bash -lc 'for p in 4 5; do python main.py --describe-phase $p --centrality n1 n3 n12 n83 --betweenness n1 n3 n12 n83 --eigenvector n1 n3 n12 n83 > centralities_phase_${p}.txt; done'
```

### Part (g)

- Use the size summary and temporal means:

```bash
python main.py --summary > summary_sizes.txt
python main.py --top-temporal betweenness --only-ids > temporal_top_betweenness_ids.txt
python main.py --top-temporal eigenvector --only-ids > temporal_top_eigenvector_ids.txt
```

### Part (h)

- Inspect comprehensive temporal rankings to surface important non‑23 actors:

```bash
python main.py --top-temporal betweenness > temporal_top_betweenness_full.txt
python main.py --top-temporal eigenvector > temporal_top_eigenvector_full.txt
```

### Part (i)

- Use directed HITS evidence to discuss in/out roles (commands above in Part (j)).

### Project (separate CLI: project.py)

- Structural metrics for all phases (CSV):

```bash
python project.py --struct-series --out struct_series.csv
```

- Top‑k rank series and Kendall τ stability:

```bash
python project.py --rank-series --metric betweenness --topk 10 --out rank_bet_top.csv
python project.py --rank-corr --metric hubs --k 10 --out hubs_kendall.txt
```

- Permutation test for an actor’s pre/post change (example):

```bash
python project.py --perm-test --actor n12 --metric betweenness --before 1-4 --after 5-11 --iters 5000 --out perm_n12_bet.txt
```

### Consolidated answers

All writeups for (f)–(j) are compiled in `answers_f_to_j.md` in this directory.


