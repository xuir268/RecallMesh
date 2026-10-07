"""Plot preregistered results; matplotlib is an optional reporting dependency."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
result = json.loads((ROOT/'benchmarks/protocols/longitudinal-results-v1.json').read_text())
rows = result['curve']
out = ROOT/'docs/assets'
out.mkdir(exist_ok=True)
with (out/'longitudinal-v1.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['session','graph_recall','adjacency_recall','bm25_recall'])
    writer.writerows((r['session'],r['graph'],r['adjacency'],r['lexical']) for r in rows)
fig, ax = plt.subplots(figsize=(8,4.5), layout='constrained')
x = [r['session'] for r in rows]
for key, label, style in [('graph','Hebbian repeated use','o-'),('adjacency','Frozen adjacency','--'),('lexical','BM25',':')]:
    ax.plot(x,[100*r[key] for r in rows],style,label=label)
ax.set(xlabel='Simulated use sessions (0 = ingestion only)',ylabel='Gold evidence recall@16 (%)',title='RecallMesh: fixed 100-question longitudinal evaluation')
ax.set_xticks(x)
ax.grid(alpha=.2)
ax.legend()
fig.savefig(out/'longitudinal-v1.png',dpi=180)
fig.savefig(out/'longitudinal-v1.svg')
