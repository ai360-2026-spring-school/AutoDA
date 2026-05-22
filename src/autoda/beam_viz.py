"""
Visualisation for BeamSearchAgent search trees.

Two outputs:
  1. ASCII text tree (always available)
  2. matplotlib figure (if matplotlib + networkx installed)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .beam import BeamSearchResult, TreeNode


# ---------------------------------------------------------------------------
# ASCII tree
# ---------------------------------------------------------------------------

def ascii_tree(result: "BeamSearchResult") -> str:
    nodes = result.all_nodes
    root = nodes["root"]
    best_id = result.best_node.node_id
    metric = "cv"

    lines: list[str] = []

    def _fmt_node(node: "TreeNode") -> str:
        flag = "★" if node.node_id == best_id else ("✗" if node.pruned else "✓")
        idea_short = (node.idea_used or "root")[:50]
        ops = "+".join(t["op"] for t in node.transforms_in_step) or "—"
        return f"{flag} [{ops}] cv={node.cv:.4f}  ← {idea_short}"

    def _walk(node: "TreeNode", prefix: str, is_last: bool) -> None:
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + _fmt_node(node))
        child_prefix = prefix + ("    " if is_last else "│   ")
        children = [nodes[cid] for cid in node.children_ids if cid in nodes]
        for i, child in enumerate(children):
            _walk(child, child_prefix, i == len(children) - 1)

    # Root line
    lines.append(f"◉ root  cv={root.cv:.4f}  (baseline)")
    children = [nodes[cid] for cid in root.children_ids if cid in nodes]
    for i, child in enumerate(children):
        _walk(child, "", i == len(children) - 1)

    lines.append("")
    lines.append(f"Best: {result.best_node.node_id}  {result.root_cv:.4f} → {result.best_cv:.4f}  (-{result.improvement_pct:.1f}%)")
    lines.append(f"Branch: {result.best_node.branch_summary()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matplotlib tree
# ---------------------------------------------------------------------------

def plot_tree(
    result: "BeamSearchResult",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """Draw search tree with matplotlib + networkx. Saves PNG if save_path given."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
    except ImportError:
        print("[beam_viz] matplotlib/networkx not installed — skipping plot")
        return

    nodes = result.all_nodes
    best_id = result.best_node.node_id
    root_cv = result.root_cv

    G = nx.DiGraph()
    for nid, n in nodes.items():
        G.add_node(nid)
        if n.parent_id:
            G.add_edge(n.parent_id, nid)

    # Hierarchical layout using graphviz if available, else spring
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = _hierarchy_pos(G, "root")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("BeamSearch — Feature Engineering Tree", fontsize=13, fontweight="bold")

    # Color nodes
    node_colors: list[str] = []
    for nid in G.nodes():
        n = nodes[nid]
        if nid == "root":
            node_colors.append("#4A90D9")   # blue
        elif nid == best_id:
            node_colors.append("#F5A623")   # gold = best
        elif n.pruned:
            node_colors.append("#E74C3C")   # red = pruned
        else:
            node_colors.append("#2ECC71")   # green = survived

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           arrowstyle="-|>", arrowsize=15,
                           edge_color="#888888", width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1800,
                           node_color=node_colors, alpha=0.92)

    # Labels: short op name + cv
    labels: dict[str, str] = {}
    for nid, n in nodes.items():
        ops = "+".join(t["op"][:8] for t in n.transforms_in_step) or "root"
        cv_str = f"{n.cv:.3f}"
        delta = root_cv - n.cv if result.best_node.node_id != "root" else 0
        labels[nid] = f"{ops}\n{cv_str}"

    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=7, font_color="white", font_weight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color="#4A90D9", label="Root"),
        mpatches.Patch(color="#F5A623", label="Best node ★"),
        mpatches.Patch(color="#2ECC71", label="Survived (beam)"),
        mpatches.Patch(color="#E74C3C", label="Pruned"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax.axis("off")

    metric_dir = "↓" if result.best_node.__class__.__name__ != "dumb" else ""
    info = (f"Baseline={root_cv:.4f}  →  Best={result.best_cv:.4f}"
            f"  (−{result.improvement_pct:.1f}%)\n"
            f"Best path: {result.best_node.branch_summary()[:80]}")
    fig.text(0.02, 0.02, info, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"[beam_viz] saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Simple hierarchical layout when graphviz is unavailable."""
    pos: dict = {}

    def _h(node, left, right, vert_loc, parent=None):
        pos[node] = ((left + right) / 2, vert_loc)
        children = list(G.successors(node))
        if children:
            step = (right - left) / len(children)
            for i, child in enumerate(children):
                _h(child, left + i * step, left + (i + 1) * step, vert_loc - vert_gap, node)

    _h(root, 0, width, vert_loc)
    return pos


def save_all(result: "BeamSearchResult") -> None:
    """Save ASCII tree + PNG to result.reports_dir."""
    text = ascii_tree(result)
    tree_path = result.reports_dir / "search_tree.txt"
    tree_path.write_text(text, encoding="utf-8")
    print(text)

    plot_tree(result, save_path=result.reports_dir / "search_tree.png")
