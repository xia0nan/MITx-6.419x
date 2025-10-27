import argparse
import csv
import os
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# --------------------------- Data loading helpers --------------------------- #

def _to_int_label(x: object) -> int:
    s = str(x).strip().strip('"').strip("'")
    if s.startswith("n") or s.startswith("N"):
        s = s[1:]
    return int(s)


def read_phase_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    try:
        df.index = df.index.map(_to_int_label)
        df.columns = df.columns.map(_to_int_label)
    except Exception:
        pass
    df = df.reindex(index=df.index, columns=df.index, fill_value=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    return df


def load_all_phases(caviar_dir: str) -> Dict[int, pd.DataFrame]:
    phases: Dict[int, pd.DataFrame] = {}
    for p in range(1, 12):
        fp = os.path.join(caviar_dir, f"phase{p}.csv")
        if os.path.exists(fp):
            phases[p] = read_phase_csv(fp)
    if not phases:
        raise FileNotFoundError(f"No phase CSVs found under {caviar_dir}")
    return phases


# --------------------------- Graph conversions ---------------------------- #

def to_undirected_binary(df: pd.DataFrame):
    import networkx as nx

    bin_df = (df > 0).astype(int)
    sym = ((bin_df.values + bin_df.values.T) > 0).astype(int)
    np.fill_diagonal(sym, 0)
    undirected = pd.DataFrame(sym, index=bin_df.index, columns=bin_df.columns)
    return nx.from_pandas_adjacency(undirected)


def to_directed_weighted(df: pd.DataFrame):
    import networkx as nx

    mat = df.copy()
    np.fill_diagonal(mat.values, 0)
    return nx.from_pandas_adjacency(mat, create_using=nx.DiGraph)


# ------------------------------ Struct metrics ----------------------------- #

def compute_struct_metrics(G) -> Dict[str, float]:
    import networkx as nx

    n = G.number_of_nodes()
    m = G.number_of_edges()
    transitivity = nx.transitivity(G) if n > 0 else 0.0
    avg_clust = nx.average_clustering(G) if n > 0 else 0.0
    try:
        assort = nx.degree_assortativity_coefficient(G) if n > 1 else 0.0
    except Exception:
        assort = float("nan")
    core_num = nx.core_number(G) if n > 0 else {}
    max_core_index = max(core_num.values()) if core_num else 0
    max_core_size = sum(1 for v in core_num.values() if v == max_core_index)
    return {
        "nodes": float(n),
        "edges": float(m),
        "transitivity": float(transitivity),
        "avg_clustering": float(avg_clust),
        "assortativity": float(assort),
        "max_core_index": float(max_core_index),
        "max_core_size": float(max_core_size),
    }


def struct_series(phases: Dict[int, pd.DataFrame]) -> List[Tuple[int, Dict[str, float]]]:
    rows: List[Tuple[int, Dict[str, float]]] = []
    for p in sorted(phases.keys()):
        G = to_undirected_binary(phases[p])
        rows.append((p, compute_struct_metrics(G)))
    return rows


# ------------------------------ Rank utilities ---------------------------- #

def undirected_scores(G, metric: str) -> Dict[int, float]:
    import networkx as nx

    if metric == "degree":
        return nx.degree_centrality(G)
    if metric == "betweenness":
        return nx.betweenness_centrality(G, normalized=True)
    if metric == "eigenvector":
        return nx.eigenvector_centrality(G, max_iter=2000, tol=1e-6)
    raise ValueError(f"Unsupported undirected metric: {metric}")


def directed_scores(G, which: str) -> Dict[int, float]:
    import networkx as nx

    hubs, auths = nx.algorithms.link_analysis.hits(
        G, max_iter=1_000_000, tol=1e-8, normalized=True
    )
    return hubs if which == "hubs" else auths


def topk_items(score: Dict[int, float], k: int) -> List[Tuple[int, float]]:
    return sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:k]


def rank_series(
    phases: Dict[int, pd.DataFrame], metric: str, topk: int
) -> List[Tuple[int, List[Tuple[int, float]]]]:
    rows = []
    for p in sorted(phases.keys()):
        if metric in {"degree", "betweenness", "eigenvector"}:
            G = to_undirected_binary(phases[p])
            score = undirected_scores(G, metric)
        elif metric in {"hubs", "authorities"}:
            Gd = to_directed_weighted(phases[p])
            score = directed_scores(Gd, metric)
        else:
            raise ValueError("Unknown metric")
        rows.append((p, topk_items(score, topk)))
    return rows


def kendall_tau_from_topk(
    a: List[Tuple[int, float]], b: List[Tuple[int, float]], k: int
) -> float:
    # Build rank maps; absent -> rank k+1
    rank_a = {node: i + 1 for i, (node, _) in enumerate(a)}
    rank_b = {node: i + 1 for i, (node, _) in enumerate(b)}
    nodes = list(set(rank_a.keys()) | set(rank_b.keys()))
    ra = [rank_a.get(n, k + 1) for n in nodes]
    rb = [rank_b.get(n, k + 1) for n in nodes]
    # Count concordant/discordant pairs (ties count as 0)
    concord = 0
    discord = 0
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            da = np.sign(ra[i] - ra[j])
            db = np.sign(rb[i] - rb[j])
            prod = da * db
            if prod > 0:
                concord += 1
            elif prod < 0:
                discord += 1
    denom = concord + discord
    return 0.0 if denom == 0 else (concord - discord) / denom


def rank_corr_series(
    phases: Dict[int, pd.DataFrame], metric: str, k: int
) -> List[Tuple[int, int, float]]:
    series = rank_series(phases, metric, k)
    out: List[Tuple[int, int, float]] = []
    for i in range(len(series) - 1):
        p1, top1 = series[i]
        p2, top2 = series[i + 1]
        tau = kendall_tau_from_topk(top1, top2, k)
        out.append((p1, p2, tau))
    return out


# ------------------------------ Permutation test --------------------------- #

def parse_range(spec: str) -> List[int]:
    a, b = spec.split("-")
    return list(range(int(a), int(b) + 1))


def actor_metric_by_phase(
    phases: Dict[int, pd.DataFrame], metric: str, actor: int
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for p in sorted(phases.keys()):
        df = phases[p]
        if metric in {"degree", "betweenness", "eigenvector"}:
            G = to_undirected_binary(df)
            score = undirected_scores(G, metric)
        elif metric in {"hubs", "authorities"}:
            Gd = to_directed_weighted(df)
            score = directed_scores(Gd, metric)
        else:
            raise ValueError("Unknown metric")
        scores[p] = float(score.get(actor, 0.0))
    return scores


def perm_test_phase_label(
    values_by_phase: Dict[int, float], before: Iterable[int], after: Iterable[int], iters: int, seed: int
) -> Tuple[float, float, float, float]:
    rng = random.Random(seed)
    before_vals = [values_by_phase.get(p, 0.0) for p in before]
    after_vals = [values_by_phase.get(p, 0.0) for p in after]
    obs = float(np.mean(after_vals) - np.mean(before_vals))
    pool = before_vals + after_vals
    n_after = len(after_vals)
    count = 0
    for _ in range(iters):
        rng.shuffle(pool)
        perm_after = pool[:n_after]
        perm_before = pool[n_after:]
        diff = float(np.mean(perm_after) - np.mean(perm_before))
        if abs(diff) >= abs(obs):
            count += 1
    pval = (count + 1) / (iters + 1)
    return pval, obs, float(np.mean(before_vals)), float(np.mean(after_vals))


# ----------------------------------- CLI ---------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="CAVIAR project utilities")
    parser.add_argument(
        "--caviar-dir", type=str, default=os.path.join(os.path.dirname(__file__), "CAVIAR")
    )
    parser.add_argument("--out", type=str, default=None)

    parser.add_argument("--struct-metrics", type=int, default=None, help="Compute struct metrics for a phase")
    parser.add_argument("--struct-series", action="store_true", help="Compute struct metrics for all phases (CSV)")

    parser.add_argument(
        "--rank-series", action="store_true", help="Emit top-k rankings per phase (CSV)"
    )
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--topk", type=int, default=10)

    parser.add_argument("--rank-corr", action="store_true", help="Kendall tau of top-k across consecutive phases")
    parser.add_argument("--k", type=int, default=10)

    parser.add_argument("--perm-test", action="store_true", help="Permutation test for actor pre/post phase ranges")
    parser.add_argument("--actor", type=str, default=None)
    parser.add_argument("--before", type=str, default=None)
    parser.add_argument("--after", type=str, default=None)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    # Plotting
    parser.add_argument("--plot-struct-series", action="store_true", help="Plot structural metrics over phases")
    parser.add_argument("--plot-hits-track", nargs="*", default=None, help="Plot hub/authority time series for given nodes (e.g., n1 n3)")
    parser.add_argument("--plot-event-bet", action="store_true", help="Bar plot of betweenness change between two phases for given nodes")
    parser.add_argument("--event-phases", nargs=2, type=int, default=[4, 5])
    parser.add_argument("--event-nodes", nargs="*", default=["n1", "n3", "n12"]) 

    args = parser.parse_args()

    phases = load_all_phases(args.caviar_dir)

    def _write(lines: Iterable[str]) -> None:
        if args.out:
            with open(args.out, "w", newline="") as f:
                for ln in lines:
                    f.write(ln)
        else:
            for ln in lines:
                print(ln, end="")

    if args.struct_metrics is not None:
        G = to_undirected_binary(phases[args.struct_metrics])
        m = compute_struct_metrics(G)
        header = [
            "phase",
            "nodes",
            "edges",
            "transitivity",
            "avg_clustering",
            "assortativity",
            "max_core_index",
            "max_core_size",
        ]
        row = [
            str(args.struct_metrics),
            str(int(m["nodes"])),
            str(int(m["edges"])),
            f"{m['transitivity']:.6f}",
            f"{m['avg_clustering']:.6f}",
            f"{m['assortativity']:.6f}",
            f"{m['max_core_index']:.0f}",
            f"{m['max_core_size']:.0f}",
        ]
        _write([",".join(header) + "\n", ",".join(row) + "\n"])
        return

    if args.struct_series:
        rows = struct_series(phases)
        header = [
            "phase",
            "nodes",
            "edges",
            "transitivity",
            "avg_clustering",
            "assortativity",
            "max_core_index",
            "max_core_size",
        ]
        out_lines = [",".join(header) + "\n"]
        for p, m in rows:
            out_lines.append(
                ",".join(
                    [
                        str(p),
                        str(int(m["nodes"])),
                        str(int(m["edges"])),
                        f"{m['transitivity']:.6f}",
                        f"{m['avg_clustering']:.6f}",
                        f"{m['assortativity']:.6f}",
                        f"{m['max_core_index']:.0f}",
                        f"{m['max_core_size']:.0f}",
                    ]
                )
                + "\n"
            )
        _write(out_lines)
        return

    if args.rank_series:
        if not args.metric:
            raise SystemExit("--metric is required for --rank-series")
        series = rank_series(phases, args.metric, args.topk)
        out_lines = ["phase,rank,node,score\n"]
        for p, items in series:
            for i, (nid, val) in enumerate(items, start=1):
                out_lines.append(f"{p},{i},n{nid},{val:.6f}\n")
        _write(out_lines)
        return

    if args.rank_corr:
        if not args.metric:
            raise SystemExit("--metric is required for --rank-corr")
        corrs = rank_corr_series(phases, args.metric, args.k)
        out_lines = ["phase1,phase2,kendall_tau\n"]
        for p1, p2, tau in corrs:
            out_lines.append(f"{p1},{p2},{tau:.6f}\n")
        _write(out_lines)
        return

    if args.perm_test:
        if not (args.actor and args.metric and args.before and args.after):
            raise SystemExit("--actor, --metric, --before, --after are required for --perm-test")
        actor_id = _to_int_label(args.actor)
        before = parse_range(args.before)
        after = parse_range(args.after)
        vals = actor_metric_by_phase(phases, args.metric, actor_id)
        pval, obs, mean_before, mean_after = perm_test_phase_label(
            vals, before, after, args.iters, args.seed
        )
        out_lines = [
            "metric,actor,before,after,obs_diff,mean_before,mean_after,iterations,p_value\n",
            f"{args.metric},n{actor_id},{args.before},{args.after},{obs:.6f},{mean_before:.6f},{mean_after:.6f},{args.iters},{pval:.6f}\n",
        ]
        _write(out_lines)
        return

    # Plotting: structural series
    if args.plot_struct_series:
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise SystemExit("matplotlib required for plotting")
        rows = struct_series(phases)
        phases_x = [p for p, _ in rows]
        metrics = {k: [] for k in [
            "transitivity",
            "avg_clustering",
            "assortativity",
            "max_core_index",
        ]}
        for _, m in rows:
            for k in metrics.keys():
                metrics[k].append(m[k])
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        ax = axes.ravel()
        ax[0].plot(phases_x, metrics["transitivity"], marker="o"); ax[0].set_title("Transitivity")
        ax[1].plot(phases_x, metrics["avg_clustering"], marker="o"); ax[1].set_title("Avg clustering")
        ax[2].plot(phases_x, metrics["assortativity"], marker="o"); ax[2].set_title("Degree assortativity")
        ax[3].plot(phases_x, metrics["max_core_index"], marker="o"); ax[3].set_title("Max k-core index")
        for a in ax:
            a.set_xlabel("Phase"); a.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = args.out or os.path.join(os.path.dirname(__file__), "fig_struct_series.png")
        fig.savefig(out_path, dpi=200)
        print(f"Saved {out_path}")
        return

    # Plotting: hits track nodes
    if args.plot_hits_track is not None and len(args.plot_hits_track) > 0:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise SystemExit("matplotlib required for plotting")
        nodes = [_to_int_label(x) for x in args.plot_hits_track]
        phases_x = sorted(phases.keys())
        series_h: Dict[int, List[float]] = {nid: [] for nid in nodes}
        series_a: Dict[int, List[float]] = {nid: [] for nid in nodes}
        for p in phases_x:
            Gd = to_directed_weighted(phases[p])
            hs = directed_scores(Gd, "hubs"); au = directed_scores(Gd, "authorities")
            for nid in nodes:
                series_h[nid].append(float(hs.get(nid, 0.0)))
                series_a[nid].append(float(au.get(nid, 0.0)))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for nid in nodes:
            ax1.plot(phases_x, series_h[nid], marker="o", label=f"n{nid}")
        ax1.set_title("HITS hubs (outgoing influence)"); ax1.grid(True, alpha=0.3); ax1.legend()
        for nid in nodes:
            ax2.plot(phases_x, series_a[nid], marker="o", label=f"n{nid}")
        ax2.set_title("HITS authorities (incoming endorsement)"); ax2.grid(True, alpha=0.3); ax2.set_xlabel("Phase")
        fig.tight_layout()
        out_path = args.out or os.path.join(os.path.dirname(__file__), "fig_hits_track.png")
        fig.savefig(out_path, dpi=200)
        print(f"Saved {out_path}")
        return

    # Plotting: event betweenness deltas
    if args.plot_event_bet:
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            raise SystemExit("matplotlib and networkx required for plotting")
        p1, p2 = args.event_phases
        nodes = [_to_int_label(x) for x in args.event_nodes]
        G1 = to_undirected_binary(phases[p1]); b1 = undirected_scores(G1, "betweenness")
        G2 = to_undirected_binary(phases[p2]); b2 = undirected_scores(G2, "betweenness")
        labels = [f"n{n}" for n in nodes]
        delta = [float(b2.get(n, 0.0) - b1.get(n, 0.0)) for n in nodes]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(labels, delta, color="#5a9bd5")
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.set_title(f"Betweenness Δ (Phase {p1} → {p2})")
        ax.set_ylabel("Δ betweenness")
        for i, v in enumerate(delta):
            ax.text(i, v + (0.01 if v >= 0 else -0.01), f"{v:.3f}", ha="center", va="bottom" if v>=0 else "top")
        fig.tight_layout()
        out_path = args.out or os.path.join(os.path.dirname(__file__), f"fig_event_bet_{p1}_{p2}.png")
        fig.savefig(out_path, dpi=200)
        print(f"Saved {out_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()


