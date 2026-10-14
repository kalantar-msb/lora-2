# Fills template.html with the figures and tables from figures.py; writes lora-residency-summary.html.
import json, pandas as pd
D = "/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2/workspace/runs/l2-9"
figs = json.load(open(f"{D}/report/figs.json")); C = pd.read_csv(f"{D}/report/cells.csv"); B = pd.read_csv(f"{D}/results_charts/base_traffic_summary.csv")
sec = lambda v: f"{v:.2f}" if v < 1 else (f"{v:.1f}" if v < 10 else f"{v:.0f}")
rows, grp = [], None
for r in C.itertuples():
    if r.group != grp:
        grp = r.group; rows.append(f'          <tr class="grp"><td colspan="11">{grp}</td></tr>')
    rows.append(f'          <tr><td class="l">{r.label}</td><td class="bv sep">{sec(r.baseline_p90)}</td><td class="tv">{sec(r.loraresidency_p90)}</td>'
                f'<td class="x">{r.baseline_p90 / r.loraresidency_p90:.1f}×</td><td class="bv sep">{sec(r.baseline_p99)}</td><td class="tv">{sec(r.loraresidency_p99)}</td>'
                f'<td class="bv sep">{sec(r.baseline_e2e)}</td><td class="tv">{sec(r.loraresidency_e2e)}</td>'
                f'<td class="bv sep">{r.baseline_tpot:.0f}</td><td class="tv">{r.loraresidency_tpot:.0f}</td><td class="sep">{r.lower} / {r.n_adapters}</td></tr>')
names = {"a1_5-base20-15rps-32tok": "15 req/s, 32 tokens", "a1_5-base20-30rps-32tok": "30 req/s, 32 tokens", "a1_5-base20-15rps-128tok": "15 req/s, 128 tokens"}
base = []
for cell, lab in names.items():
    g = B[(B.cell == cell) & (B.requests == "base model")].set_index("arm")
    v = lambda m, a: f"{g.loc[a, m]:,.0f}"
    base.append(f'            <tr><td class="l">{lab}</td><td class="bv sep">{v("ttft_p50", "baseline")}</td><td class="tv">{v("ttft_p50", "loraresidency")}</td>'
                f'<td class="bv sep">{v("ttft_p99", "baseline")}</td><td class="tv">{v("ttft_p99", "loraresidency")}</td>'
                f'<td class="bv sep">{v("tpot_p50", "baseline")}</td><td class="tv">{v("tpot_p50", "loraresidency")}</td></tr>')
html = open(f"{D}/report/template.html").read().replace("{{TABLE}}", "\n".join(rows)).replace("{{BASE}}", "\n".join(base))
for k, v in figs.items(): html = html.replace("{{" + k.upper() + "}}", v)
assert "{{" not in html
open(f"{D}/report/lora-residency-summary.html", "w").write(html); print(f"wrote {len(html)/1e6:.2f} MB")
