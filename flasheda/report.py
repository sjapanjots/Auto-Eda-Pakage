"""
flasheda.report
~~~~~~~~~~~~~~~
EDAReport: the single object returned by flasheda.analyze().
Supports:
  - .show()         → rich console table
  - .to_html()      → self-contained HTML report
  - .to_dict()      → plain dict
  - .to_json()      → JSON string
  - .save_html()    → write HTML to disk
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class EDAReport:
    overview: Dict[str, Any]
    missing: Dict[str, Any]
    numeric: Dict[str, Any]
    categorical: Dict[str, Any]
    correlation: Dict[str, Any]
    sample_size: int
    elapsed_seconds: float
    original_shape: tuple
    warnings: list[str] = field(default_factory=list)

    # ──────────────────────────────────────────────────────────────────────
    # Console output
    # ──────────────────────────────────────────────────────────────────────

    def show(self) -> None:
        """Print a rich summary to the console."""
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from rich.panel import Panel

        console = Console()

        # ── Header ──
        orig_rows = self.original_shape[0]
        ncols = self.original_shape[1]
        console.print(
            Panel(
                f"[bold cyan]FlashEDA Report[/bold cyan]\n"
                f"Dataset: [green]{orig_rows:,}[/green] rows × [green]{ncols}[/green] cols  "
                f"| Analysed on [yellow]{self.sample_size:,}[/yellow] sampled rows  "
                f"| ⏱  [white]{self.elapsed_seconds:.2f}s[/white]",
                expand=False,
            )
        )

        # ── Warnings ──
        if self.warnings:
            for w in self.warnings:
                console.print(f"  [bold yellow]⚠  {w}[/bold yellow]")
            console.print()

        # ── Missing values ──
        miss = self.missing
        console.print(f"[bold]Missing values[/bold]  overall: [red]{miss['overall_null_pct']}%[/red]  "
                      f"| rows with any null: [red]{miss['rows_with_any_null_pct']}%[/red]")
        if miss["critical_columns"]:
            console.print(f"  Critical (>50% null): [red]{', '.join(miss['critical_columns'])}[/red]")
        if miss["warn_columns"]:
            console.print(f"  Warn    (>20% null): [yellow]{', '.join(miss['warn_columns'])}[/yellow]")
        console.print()

        # ── Numeric summary table ──
        num_cols = self.numeric.get("numeric_col_names", [])
        if num_cols:
            t = Table(title="Numeric columns", box=box.SIMPLE, show_lines=False)
            for h in ["Column", "Mean", "Std", "Min", "Median", "Max", "Outlier %", "Skew"]:
                t.add_column(h, style="white")
            for col in num_cols:
                s = self.numeric["columns"].get(col, {})
                if "error" in s:
                    continue
                flag = "[yellow]⚠[/yellow] " if s.get("skewed") else ""
                t.add_row(
                    col,
                    str(s.get("mean", "")),
                    str(s.get("std", "")),
                    str(s.get("min", "")),
                    str(s.get("median", "")),
                    str(s.get("max", "")),
                    f"{s.get('outlier_pct', 0)}%",
                    f"{flag}{s.get('skewness', '')}",
                )
            console.print(t)

        # ── Categorical summary table ──
        cat_cols = self.categorical.get("categorical_col_names", [])
        if cat_cols:
            t = Table(title="Categorical columns", box=box.SIMPLE, show_lines=False)
            for h in ["Column", "Unique", "Cardinality ratio", "Mode", "Mode %", "Flag"]:
                t.add_column(h)
            for col in cat_cols:
                s = self.categorical["columns"].get(col, {})
                if "error" in s:
                    continue
                flag = ""
                if s.get("likely_id_column"):
                    flag = "[red]likely ID[/red]"
                elif s.get("high_cardinality"):
                    flag = "[yellow]high cardinality[/yellow]"
                t.add_row(
                    col,
                    str(s.get("unique_count", "")),
                    str(s.get("cardinality_ratio", "")),
                    str(s.get("mode", ""))[:30],
                    f"{s.get('mode_pct', '')}%",
                    flag,
                )
            console.print(t)

        # ── Strong correlations ──
        strong = self.correlation.get("numeric_strong_pairs", [])
        if strong:
            console.print("[bold]Strong numeric correlations (|r| ≥ 0.8)[/bold]")
            for pair in strong[:10]:
                console.print(f"  {pair['col_a']} ↔ {pair['col_b']}  r={pair['pearson_r']}")
            console.print()

    # ──────────────────────────────────────────────────────────────────────
    # HTML report
    # ──────────────────────────────────────────────────────────────────────

    def to_html(self) -> str:
        """Return a self-contained HTML report string."""
        return _render_html(self)

    def save_html(self, path: str = "flasheda_report.html") -> Path:
        """Save the HTML report to disk and return the path."""
        p = Path(path)
        p.write_text(self.to_html(), encoding="utf-8")
        return p

    # ──────────────────────────────────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overview": self.overview,
            "missing": self.missing,
            "numeric": self.numeric,
            "categorical": self.categorical,
            "correlation": self.correlation,
            "sample_size": self.sample_size,
            "elapsed_seconds": self.elapsed_seconds,
            "original_shape": list(self.original_shape),
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __repr__(self) -> str:
        rows, cols = self.original_shape
        return (
            f"EDAReport(rows={rows:,}, cols={cols}, "
            f"sample={self.sample_size:,}, elapsed={self.elapsed_seconds:.2f}s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HTML renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_html(r: EDAReport) -> str:
    num_stats = r.numeric.get("numeric_col_names", [])
    cat_stats = r.categorical.get("categorical_col_names", [])
    strong_corr = r.correlation.get("numeric_strong_pairs", [])
    miss_per_col = r.missing.get("per_column", {})

    # Build numeric rows
    num_rows_html = ""
    for col in num_stats:
        s = r.numeric["columns"].get(col, {})
        if "error" in s:
            continue
        skew_badge = '<span class="badge warn">skewed</span>' if s.get("skewed") else ""
        out_badge = '<span class="badge warn">outliers</span>' if s.get("outlier_pct", 0) > 5 else ""
        num_rows_html += f"""
        <tr>
          <td><strong>{col}</strong></td>
          <td>{s.get('mean', '')}</td><td>{s.get('std', '')}</td>
          <td>{s.get('min', '')}</td><td>{s.get('median', '')}</td><td>{s.get('max', '')}</td>
          <td>{s.get('outlier_pct', '')}%</td>
          <td>{s.get('skewness', '')} {skew_badge}</td>
          <td>{out_badge}</td>
        </tr>"""

    # Build categorical rows
    cat_rows_html = ""
    for col in cat_stats:
        s = r.categorical["columns"].get(col, {})
        if "error" in s:
            continue
        flag = ""
        if s.get("likely_id_column"):
            flag = '<span class="badge danger">likely ID</span>'
        elif s.get("high_cardinality"):
            flag = '<span class="badge warn">high cardinality</span>'
        top = s.get("top_values", [])[:5]
        top_str = " · ".join(f"{t['value']} ({t['pct']}%)" for t in top)
        cat_rows_html += f"""
        <tr>
          <td><strong>{col}</strong></td>
          <td>{s.get('unique_count', '')}</td>
          <td>{s.get('cardinality_ratio', '')}</td>
          <td>{s.get('mode', '')[:40]}</td>
          <td>{s.get('mode_pct', '')}%</td>
          <td class="muted">{top_str[:80]}</td>
          <td>{flag}</td>
        </tr>"""

    # Missing table
    miss_rows_html = ""
    for col, info in miss_per_col.items():
        sev = info.get("severity", "ok")
        color = {"ok": "#27ae60", "warn": "#f39c12", "critical": "#e74c3c"}.get(sev, "#888")
        bar_w = min(100, int(info.get("null_pct", 0)))
        miss_rows_html += f"""
        <tr>
          <td>{col}</td>
          <td>{info.get('null_count', 0)}</td>
          <td>
            <div class="bar-bg">
              <div class="bar-fill" style="width:{bar_w}%;background:{color}"></div>
            </div>
            {info.get('null_pct', 0)}%
          </td>
          <td><span style="color:{color};font-weight:500">{sev}</span></td>
        </tr>"""

    # Correlation rows
    corr_html = ""
    for p in strong_corr[:15]:
        corr_html += f"<tr><td>{p['col_a']}</td><td>{p['col_b']}</td><td>{p['pearson_r']}</td></tr>"

    # Warnings
    warn_html = ""
    for w in r.warnings:
        warn_html += f'<div class="warn-banner">⚠  {w}</div>'

    orig_rows, orig_cols = r.original_shape

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FlashEDA Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #f5f7fa; color: #222; font-size: 14px; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px; }}
  header {{ background: #1a1a2e; color: #fff; padding: 24px 32px; border-radius: 10px; margin-bottom: 24px; }}
  header h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 6px; }}
  header .meta {{ font-size: 13px; color: #aac; }}
  .card {{ background: #fff; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.07); }}
  .card h2 {{ font-size: 15px; font-weight: 600; margin-bottom: 14px; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 6px 10px; background: #f0f2f5; font-weight: 500; }}
  td {{ padding: 6px 10px; border-top: 1px solid #f0f0f0; vertical-align: middle; }}
  .badge {{ display: inline-block; padding: 2px 7px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
  .badge.warn {{ background: #fff3cd; color: #856404; }}
  .badge.danger {{ background: #f8d7da; color: #842029; }}
  .muted {{ color: #888; font-size: 12px; }}
  .bar-bg {{ display: inline-block; width: 80px; height: 8px; background: #eee; border-radius: 4px; vertical-align: middle; margin-right: 6px; }}
  .bar-fill {{ height: 8px; border-radius: 4px; }}
  .warn-banner {{ background: #fff3cd; color: #856404; padding: 10px 14px; border-radius: 8px; margin-bottom: 10px; font-size: 13px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 4px; }}
  .stat-box {{ background: #f8f9fc; border-radius: 8px; padding: 14px 16px; }}
  .stat-box .label {{ font-size: 11px; color: #888; margin-bottom: 4px; }}
  .stat-box .value {{ font-size: 20px; font-weight: 600; color: #1a1a2e; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>⚡ FlashEDA Report</h1>
    <div class="meta">
      {orig_rows:,} rows × {orig_cols} columns &nbsp;|&nbsp;
      Analysed on {r.sample_size:,} sampled rows &nbsp;|&nbsp;
      Completed in {r.elapsed_seconds:.2f}s
    </div>
  </header>

  {warn_html}

  <div class="card">
    <h2>Dataset overview</h2>
    <div class="stats-grid">
      <div class="stat-box"><div class="label">Total rows</div><div class="value">{orig_rows:,}</div></div>
      <div class="stat-box"><div class="label">Columns</div><div class="value">{orig_cols}</div></div>
      <div class="stat-box"><div class="label">Sample size</div><div class="value">{r.sample_size:,}</div></div>
      <div class="stat-box"><div class="label">Analysis time</div><div class="value">{r.elapsed_seconds:.2f}s</div></div>
      <div class="stat-box"><div class="label">Overall null %</div><div class="value">{r.missing.get('overall_null_pct', 0)}%</div></div>
      <div class="stat-box"><div class="label">Duplicate rows (sample)</div><div class="value">{r.overview.get('duplicate_pct_in_sample', 0)}%</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Missing values</h2>
    <table>
      <thead><tr><th>Column</th><th>Null count</th><th>Null %</th><th>Status</th></tr></thead>
      <tbody>{miss_rows_html}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Numeric columns</h2>
    <table>
      <thead><tr><th>Column</th><th>Mean</th><th>Std</th><th>Min</th><th>Median</th><th>Max</th><th>Outlier %</th><th>Skewness</th><th>Flags</th></tr></thead>
      <tbody>{num_rows_html}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Categorical columns</h2>
    <table>
      <thead><tr><th>Column</th><th>Unique</th><th>Cardinality ratio</th><th>Mode</th><th>Mode %</th><th>Top values</th><th>Flags</th></tr></thead>
      <tbody>{cat_rows_html}</tbody>
    </table>
  </div>

  {"" if not corr_html else f'''
  <div class="card">
    <h2>Strong correlations (|r| ≥ 0.8)</h2>
    <table>
      <thead><tr><th>Column A</th><th>Column B</th><th>Pearson r</th></tr></thead>
      <tbody>{corr_html}</tbody>
    </table>
  </div>'''}

  <p class="muted" style="text-align:center;margin-top:16px">Generated by FlashEDA — constant-time EDA for any dataset size</p>
</div>
</body>
</html>"""