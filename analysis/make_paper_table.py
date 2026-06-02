#!/usr/bin/env python3
"""
Generate a paper-ready table from PR analysis CSV outputs.

Reads `<prefix>_last_layer_summary.csv` and prints both Markdown and LaTeX
tables to stdout; optionally saves them to files.

Usage:
    python make_paper_table.py --prefix outputs/pr_summary \
        --save-md outputs/paper_table.md --save-tex outputs/paper_table.tex
"""

import argparse
from pathlib import Path


def load_last_layer_summary(path: Path):
    rows = []
    with open(path, "r") as f:
        header = next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            eps, thr, cnt, prop = line.split(",")
            rows.append({
                'epsilon': float(eps),
                'pr_threshold': float(thr),
                'count': int(cnt),
                'proportion': float(prop),
            })
    return rows


def to_markdown(rows):
    # Columns: epsilon, PR threshold, count, proportion
    lines = []
    lines.append("| Allowed Loss (ε) | PR Threshold | Count | Proportion |")
    lines.append("|:-----------------:|:------------:|------:|-----------:|")
    for r in rows:
        lines.append(
            f"| {r['epsilon']:.2%} | {r['pr_threshold']:.6f} | {r['count']} | {r['proportion']:.2%} |")
    return "\n".join(lines)


def to_latex(rows):
    # A compact LaTeX tabular (booktabs recommended in the paper preamble)
    lines = []
    lines.append(r"\begin{tabular}{r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Allowed Loss ($\varepsilon$) & PR Threshold & Count & Proportion " + "\\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(
            f"{r['epsilon']:.0%} & {r['pr_threshold']:.6f} & {r['count']} & {r['proportion']:.2%}" + " \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Create paper table from PR analysis outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prefix", type=Path, required=True,
                        help="Prefix used by pr_analysis.py --save-csv")
    parser.add_argument("--save-md", type=Path, default=None,
                        help="Optional path to save Markdown table")
    parser.add_argument("--save-tex", type=Path, default=None,
                        help="Optional path to save LaTeX table")
    args = parser.parse_args()

    summary_path = args.prefix.with_name(args.prefix.name + "_last_layer_summary.csv")
    rows = load_last_layer_summary(summary_path)

    md = to_markdown(rows)
    tex = to_latex(rows)

    print("\nMarkdown table:\n")
    print(md)
    print("\nLaTeX table:\n")
    print(tex)

    if args.save_md is not None:
        args.save_md.parent.mkdir(parents=True, exist_ok=True)
        args.save_md.write_text(md)
        print("\nSaved Markdown:", str(args.save_md))

    if args.save_tex is not None:
        args.save_tex.parent.mkdir(parents=True, exist_ok=True)
        args.save_tex.write_text(tex)
        print("Saved LaTeX:", str(args.save_tex))


if __name__ == "__main__":
    main()
