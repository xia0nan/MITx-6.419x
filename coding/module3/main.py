import os
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


def read_phase_csv(csv_path: str) -> pd.DataFrame:
    """Read a CAVIAR phase CSV into a numeric pandas DataFrame.

    The files encode an adjacency matrix of counts. We:
    - use the first column as index (player ids)
    - ensure numeric dtype
    - return the raw (possibly asymmetric) matrix
    """
    df = pd.read_csv(csv_path, index_col=0)
    # Normalize labels so index and columns are the same integer space
    def _to_int_label(x: object) -> int:
        s = str(x).strip().strip('"').strip("'")
        if s.startswith("n"):
            s = s[1:]
        return int(s)

    try:
        df.index = df.index.map(_to_int_label)
        df.columns = df.columns.map(_to_int_label)
    except Exception:
        # Best effort: if conversion fails for any reason, leave labels as-is
        pass

    # Ensure the matrix has identical row/column labels and order
    try:
        df = df.reindex(index=df.index, columns=df.index, fill_value=0)
    except Exception:
        pass
    # Ensure numeric type; some CSVs may parse as object due to quotes in headers
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    return df


def to_undirected_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Convert directed weighted adjacency to undirected binary.

    - We consider an undirected tie present if there is communication in
      either direction between two actors during the phase.
    - We binarize the matrix (any positive becomes 1).
    - We symmetrize using logical OR with the transpose.
    - We zero out self-loops.
    """
    bin_df = (df > 0).astype(int)
    sym = ((bin_df.values + bin_df.values.T) > 0).astype(int)
    np.fill_diagonal(sym, 0)
    undirected = pd.DataFrame(sym, index=bin_df.index, columns=bin_df.columns)
    return undirected


def to_directed_weighted(df: pd.DataFrame):
    """Convert the raw adjacency to a directed NetworkX DiGraph with weights."""
    import networkx as nx

    # Ensure no self loops and keep weights
    mat = df.copy()
    np.fill_diagonal(mat.values, 0)
    G = nx.from_pandas_adjacency(mat, create_using=nx.DiGraph)
    return G


def count_nodes_edges(undirected_binary: pd.DataFrame) -> Tuple[int, int, int]:
    """Return (total_nodes, active_nodes, undirected_edges).

    - total_nodes: dimension of the matrix (includes isolates)
    - active_nodes: nodes with degree > 0 in the undirected graph
    - undirected_edges: number of undirected edges (above-diagonal ones)
    """
    total_nodes = undirected_binary.shape[0]
    degrees = undirected_binary.values.sum(axis=1)
    active_nodes = int((degrees > 0).sum())
    # Count edges from upper triangle only
    edges = int(np.triu(undirected_binary.values, k=1).sum())
    return total_nodes, active_nodes, edges


def to_networkx_graph(undirected_binary: pd.DataFrame):
    try:
        import networkx as nx
    except ImportError:  # pragma: no cover - runtime path
        raise
    return nx.from_pandas_adjacency(undirected_binary)


def parse_node_label(label: str) -> int:
    s = str(label).strip().lower()
    if s.startswith("n") or s.startswith("m"):
        s = s[1:]
    return int(s)


# The 23 players under investigation (by id only)
SERERO_ORG_23 = [
    1, 3, 83, 86, 85, 6, 11, 88, 106, 89, 84, 5, 8, 76, 77, 87, 82, 96, 12,
    17, 80, 33, 16,
]


def _compute_centrality(G, metric: str) -> Dict[int, float]:
    import networkx as nx  # local import to keep optional

    if metric == "degree":
        return nx.degree_centrality(G)
    if metric == "betweenness":
        return nx.betweenness_centrality(G, normalized=True)
    if metric == "eigenvector":
        try:
            return nx.eigenvector_centrality(G, max_iter=2000, tol=1e-6)
        except Exception:
            # Fallback for non-convergence
            try:
                return nx.eigenvector_centrality_numpy(G)
            except Exception:
                return {n: 0.0 for n in G.nodes}
    raise ValueError(f"Unknown metric: {metric}")


def compute_temporal_means(
    phases: Dict[int, pd.DataFrame],
    metric: str,
    node_filter: List[int] | None = None,
) -> Dict[int, float]:
    """Mean centrality over all 11 phases, filling 0 for missing nodes/phases."""
    import networkx as nx  # local import

    # Define the universe of nodes to report
    if node_filter is not None:
        all_nodes = list(dict.fromkeys(int(n) for n in node_filter))
    else:
        all_nodes_set = set()
        for df in phases.values():
            all_nodes_set.update(int(i) for i in df.index)
        all_nodes = sorted(all_nodes_set)

    totals: Dict[int, float] = {n: 0.0 for n in all_nodes}
    num_phases = 11
    for phase in sorted(phases.keys()):
        undirected = to_undirected_binary(phases[phase])
        G = nx.from_pandas_adjacency(undirected)
        c = _compute_centrality(G, metric)
        for n in all_nodes:
            totals[n] += float(c.get(n, 0.0))
    return {n: totals[n] / float(num_phases) for n in all_nodes}


def load_all_phases(base_dir: str) -> Dict[int, pd.DataFrame]:
    phases: Dict[int, pd.DataFrame] = {}
    for phase in range(1, 12):
        csv_path = os.path.join(base_dir, f"phase{phase}.csv")
        if not os.path.exists(csv_path):
            continue
        df = read_phase_csv(csv_path)
        phases[phase] = df
    return phases


def summarize_phases(phases: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Tuple[int, int, int, int]] = []
    for phase in sorted(phases.keys()):
        undirected = to_undirected_binary(phases[phase])
        total_nodes, active_nodes, edges = count_nodes_edges(undirected)
        rows.append((phase, total_nodes, active_nodes, edges))
    result = pd.DataFrame(
        rows, columns=["phase", "nodes_total", "nodes_active", "edges"]
    ).set_index("phase")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CAVIAR network exploration")
    parser.add_argument(
        "--plot-phase",
        type=int,
        default=None,
        help="If set, draw and save the specified phase graph",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default=None,
        help="Output image path for --plot-phase (default: ./phase{p}.png)",
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="Draw node labels (e.g., n1, n2, ...) on the plot",
    )
    parser.add_argument(
        "--plot-pair",
        type=int,
        default=None,
        help="Plot consecutive phases X and X+1 side-by-side (use with --with-labels)",
    )
    parser.add_argument(
        "--plot-pairs-all",
        action="store_true",
        help="Export all consecutive phase pairs (1-2, 2-3, ..., 10-11) to ./pairs/",
    )
    parser.add_argument(
        "--describe-phase",
        type=int,
        default=None,
        help="If set, print structural stats for the specified phase",
    )
    parser.add_argument(
        "--centrality",
        nargs="*",
        default=None,
        help="Compute degree centrality for given nodes (e.g., n1 n3 n12 n83) at --describe-phase",
    )
    parser.add_argument(
        "--betweenness",
        nargs="*",
        default=None,
        help="Compute betweenness centrality for given nodes at --describe-phase",
    )
    parser.add_argument(
        "--eigenvector",
        nargs="*",
        default=None,
        help="Compute eigenvector centrality for given nodes at --describe-phase",
    )
    parser.add_argument(
        "--top-temporal",
        type=str,
        choices=["betweenness", "eigenvector", "degree"],
        default=None,
        help="Compute per-node mean centrality across 11 phases and print top 3",
    )
    parser.add_argument(
        "--players",
        nargs="*",
        default=None,
        help="Restrict temporal means to these players (e.g., n1 n3 ...). Default: 23 suspects",
    )
    parser.add_argument(
        "--suspects-only",
        action="store_true",
        help="Restrict temporal means to the 23 suspects (default is ALL actors)",
    )
    parser.add_argument(
        "--only-ids",
        action="store_true",
        help="When used with --top-temporal, print only the top-3 node IDs (space-separated)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print per-phase size summary and key phases (2,6,10)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print structural stats for --describe-phase (components, degree summary)",
    )
    parser.add_argument(
        "--hits-phase",
        type=int,
        default=None,
        help="Compute HITS (hubs/authorities) for a directed graph at this phase",
    )
    parser.add_argument(
        "--hits-top",
        type=int,
        default=10,
        help="Top-K to print for --hits-phase",
    )
    parser.add_argument(
        "--hits-track",
        nargs="*",
        default=None,
        help="Track hubs/authorities for specified nodes (e.g., n1 n3) across all phases",
    )

    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "CAVIAR")
    phases = load_all_phases(base_dir)
    if not phases:
        raise FileNotFoundError(
            f"No CAVIAR CSV files found under: {base_dir}. Expected phase1.csv .. phase11.csv"
        )

    if args.summary:
        summary = summarize_phases(phases)
        print("CAVIAR network size per phase (undirected, binary):")
        print(summary.to_string())
        for p in [2, 6, 10]:
            if p in summary.index:
                row = summary.loc[p]
                print(
                    f"Phase {p}: nodes_total={row.nodes_total}, nodes_active={row.nodes_active}, edges={row.edges}"
                )

    # Optional plotting of a specific phase
    if args.plot_phase is not None and args.plot_phase in phases:
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            undirected = to_undirected_binary(phases[args.plot_phase])
            G = nx.from_pandas_adjacency(undirected)
            try:
                from networkx.drawing.nx_agraph import graphviz_layout

                pos = graphviz_layout(G)
            except Exception:
                # Fallback layout if pygraphviz/graphviz is not available
                pos = nx.spring_layout(G, seed=42)

            plt.figure(figsize=(8.0, 5.6))
            labels = None
            if args.with_labels:
                labels = {n: f"n{n}" for n in G.nodes}
            nx.draw(
                G,
                pos=pos,
                with_labels=args.with_labels,
                labels=labels,
                font_size=7,
                node_size=260,
                node_color="#cd69cd",
                edge_color="#777777",
                width=0.8,
            )
            plt.title(f"CAVIAR Phase {args.plot_phase}")
            plt.tight_layout()

            out_path = (
                args.plot_out
                if args.plot_out is not None
                else os.path.join(os.path.dirname(__file__), f"phase{args.plot_phase}.png")
            )
            plt.savefig(out_path, dpi=220)
            plt.close()
            print(f"Saved phase {args.plot_phase} plot to: {out_path}")
        except ImportError as e:
            print(
                "Plotting requires networkx and matplotlib. Install them and retry. Error:\n" + str(e)
            )

    # Optional textual description for a phase
    if args.describe_phase is not None and args.describe_phase in phases:
        try:
            import networkx as nx

            undirected = to_undirected_binary(phases[args.describe_phase])
            G = nx.from_pandas_adjacency(undirected)
            degrees = dict(G.degree())
            deg_values = np.array(list(degrees.values()))
            if args.stats:
                components = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
                hub_node, hub_degree = max(degrees.items(), key=lambda kv: kv[1]) if degrees else (None, 0)
                print(
                    "\nPhase {} structural stats:".format(args.describe_phase)
                )
                print(f"  nodes (active): {G.number_of_nodes()}")
                print(f"  edges: {G.number_of_edges()}")
                print(f"  components: {len(components)} | sizes: {components[:8]}")
                print(
                    f"  degree: min={deg_values.min() if deg_values.size else 0}, "
                    f"median={np.median(deg_values) if deg_values.size else 0}, "
                    f"max={deg_values.max() if deg_values.size else 0} (node {hub_node})"
                )
                leaves = int((deg_values == 1).sum())
                print(f"  leaves (deg=1): {leaves}")
            # Optional specific node centralities
            if args.centrality:
                requested: List[int] = [parse_node_label(x) for x in args.centrality]
                dc = nx.degree_centrality(G)
                print("  requested normalized degree centralities:")
                for node_id in requested:
                    val = dc.get(node_id, 0.0)
                    print(f"    n{node_id}: {val:.6f}")
            if args.betweenness:
                requested_b: List[int] = [parse_node_label(x) for x in args.betweenness]
                bc = nx.betweenness_centrality(G, normalized=True)
                print("  requested normalized betweenness centralities:")
                for node_id in requested_b:
                    val = bc.get(node_id, 0.0)
                    print(f"    n{node_id}: {val:.6f}")
            if args.eigenvector:
                requested_e: List[int] = [parse_node_label(x) for x in args.eigenvector]
                ec = nx.eigenvector_centrality(G, max_iter=2000, tol=1e-06)
                # Optional normalization check (L2 norm == 1)
                # l2_norm = float(np.sqrt(sum(v * v for v in ec.values())))
                print("  requested eigenvector centralities (L2-normalized):")
                for node_id in requested_e:
                    val = ec.get(node_id, 0.0)
                    print(f"    n{node_id}: {val:.6f}")
        except ImportError as e:
            print("Describing a phase requires networkx. Install it and retry.\n" + str(e))

    # Plot a consecutive pair (X, X+1)
    if args.plot_pair is not None and (args.plot_pair + 1) in phases:
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            from pathlib import Path

            def _build_graph(p: int):
                undirected = to_undirected_binary(phases[p])
                return nx.from_pandas_adjacency(undirected)

            Gx = _build_graph(args.plot_pair)
            Gxp1 = _build_graph(args.plot_pair + 1)

            try:
                from networkx.drawing.nx_agraph import graphviz_layout

                pos_x = graphviz_layout(Gx)
                pos_y = graphviz_layout(Gxp1)
            except Exception:
                pos_x = nx.spring_layout(Gx, seed=42)
                pos_y = nx.spring_layout(Gxp1, seed=42)

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            for ax, G, pos, title in [
                (axes[0], Gx, pos_x, f"Phase {args.plot_pair}"),
                (axes[1], Gxp1, pos_y, f"Phase {args.plot_pair + 1}"),
            ]:
                labels = {n: f"n{n}" for n in G.nodes} if args.with_labels else None
                nx.draw(
                    G,
                    pos=pos,
                    ax=ax,
                    with_labels=args.with_labels,
                    labels=labels,
                    node_size=260,
                    font_size=7,
                    node_color="#a1e46d",
                    edge_color="#777777",
                    width=0.8,
                )
                ax.set_title(title)
            fig.tight_layout()
            out_dir = Path(os.path.dirname(__file__))
            out_path = out_dir / f"pair_{args.plot_pair}_{args.plot_pair + 1}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"Saved pair plot to: {out_path}")
        except ImportError as e:
            print("Plotting requires networkx and matplotlib.\n" + str(e))

    # Export all pairs
    if args.plot_pairs_all:
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            from pathlib import Path

            out_dir = Path(os.path.dirname(__file__)) / "pairs"
            out_dir.mkdir(parents=True, exist_ok=True)

            for p in range(1, 11):
                if p not in phases or (p + 1) not in phases:
                    continue
                und_x = to_undirected_binary(phases[p])
                und_y = to_undirected_binary(phases[p + 1])
                Gx = nx.from_pandas_adjacency(und_x)
                Gy = nx.from_pandas_adjacency(und_y)
                try:
                    from networkx.drawing.nx_agraph import graphviz_layout

                    pos_x = graphviz_layout(Gx)
                    pos_y = graphviz_layout(Gy)
                except Exception:
                    pos_x = nx.spring_layout(Gx, seed=42)
                    pos_y = nx.spring_layout(Gy, seed=42)

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                for ax, G, pos, title in [
                    (axes[0], Gx, pos_x, f"Phase {p}"),
                    (axes[1], Gy, pos_y, f"Phase {p + 1}"),
                ]:
                    labels = {n: f"n{n}" for n in G.nodes} if args.with_labels else None
                    nx.draw(
                        G,
                        pos=pos,
                        ax=ax,
                        with_labels=args.with_labels,
                        labels=labels,
                        node_size=260,
                        font_size=7,
                        node_color="#a1e46d",
                        edge_color="#777777",
                        width=0.8,
                    )
                    ax.set_title(title)
                fig.tight_layout()
                out_path = out_dir / f"pair_{p}_{p + 1}.png"
                fig.savefig(out_path, dpi=200)
                plt.close(fig)
                print(f"Saved: {out_path}")
        except ImportError as e:
            print("Plotting requires networkx and matplotlib.\n" + str(e))

    # Temporal means across phases (Q5)
    if args.top_temporal is not None:
        # Determine which node set to evaluate
        nodes = None
        if args.players:
            nodes = [parse_node_label(x) for x in args.players]
        elif args.suspects_only:
            nodes = SERERO_ORG_23

        means = compute_temporal_means(phases, args.top_temporal, node_filter=nodes)
        top3 = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_ids = [str(nid) for nid, _ in top3]
        if args.only_ids:
            # Print just the integer indices, space-separated, ready to paste in the grader
            print(" ".join(top_ids))
        else:
            print(f"Top 3 mean {args.top_temporal} centrality across phases:")
            for rank, (nid, val) in enumerate(top3, start=1):
                print(f"  {rank}. n{nid}: {val:.6f}")
            print("Top 3 IDs (copy for grader): " + " ".join(top_ids))

    # HITS for directed graphs (Part j)
    if args.hits_phase is not None and args.hits_phase in phases:
        try:
            import networkx as nx

            df = phases[args.hits_phase]
            Gd = to_directed_weighted(df)
            hubs, auths = nx.algorithms.link_analysis.hits(
                Gd, max_iter=1_000_000, tol=1e-08, normalized=True
            )
            top_h = sorted(hubs.items(), key=lambda kv: kv[1], reverse=True)[: args.hits_top]
            top_a = sorted(auths.items(), key=lambda kv: kv[1], reverse=True)[: args.hits_top]
            print(f"Phase {args.hits_phase} - Top hubs:")
            for nid, val in top_h:
                print(f"  n{nid}: {val:.6f}")
            print(f"Phase {args.hits_phase} - Top authorities:")
            for nid, val in top_a:
                print(f"  n{nid}: {val:.6f}")
        except ImportError as e:
            print("HITS requires networkx.\n" + str(e))

    if args.hits_track is not None:
        try:
            import networkx as nx

            track_nodes = [parse_node_label(x) for x in args.hits_track] if args.hits_track else []
            print("node,phase,hub,authority")
            for p in sorted(phases.keys()):
                df = phases[p]
                Gd = to_directed_weighted(df)
                hubs, auths = nx.algorithms.link_analysis.hits(
                    Gd, max_iter=1_000_000, tol=1e-08, normalized=True
                )
                for nid in track_nodes:
                    print(f"n{nid},{p},{hubs.get(nid, 0.0):.6f},{auths.get(nid, 0.0):.6f}")
        except ImportError as e:
            print("HITS requires networkx.\n" + str(e))


if __name__ == "__main__":
    main()


