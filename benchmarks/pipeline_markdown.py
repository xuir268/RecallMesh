"""Produce a readable report from the immutable attempt records."""
from pathlib import Path
import json
OUT=Path('benchmarks/results/pipeline')
summary=json.loads((OUT/'summary.json').read_text())
lines=['# Context-memory answer pipeline pilot','',
'Bounded follow-up retrieval, joint evidence selection, whole-record context compression, and a model support check are implemented. The CLI is `assoc-memory`.','',
'Twelve fixed questions: six new generated instances across alias, arithmetic, and temporal families; six previously untested [public LoCoMo](https://github.com/snap-research/locomo/blob/main/data/locomo10.json) questions, two each from categories 1, 2, and 4. This small pilot does not establish population accuracy or a universally best small model.','',
'Generated score is exact answer accuracy; natural score is the saved upstream-style answer F1. Failed formats count as failures. Account unavailability is reported separately. Wider one-call retrieval is a control, with a larger final evidence budget. Bounded and single have the same maximum final record count, but bounded sees more candidates and uses more model calls.','',
'| Backend | Suite | Arm | Answers / attempts | Unavailable | Score on available attempts | Median seconds | Mean input tokens across calls |',
'|---|---|---|---:|---:|---:|---:|---:|']
for model,groups in summary.items():
 if model=='claude' and 'claude-resumed' in summary:continue
 for key,g in groups.items():
  suite,arm=key.split('-',1)
  def number(x,digits=2):return '—' if x is None else f'{x:.{digits}f}'
  lines.append(f"| {model} | {suite} | {arm} | {g['completed']}/{g['attempted']} | {g['unavailable']} | {number(g['mean_score_available']*100 if g['mean_score_available'] is not None else None)}% | {number(g['median_seconds_success'])} | {number(g['mean_input_tokens_all_calls'],0)} |")
lines+=['','## Findings','',
'Astra: generated exact answers improved from 2/6 to 5/6. Expanding the one-call candidate set stayed at 2/6. Four bounded generated questions issued follow-up searches. This supports the mechanism on these instances, not a general reasoning improvement.','',
'Astra: natural mean F1 stayed at 45.18% with bounded retrieval; the wider one-call control reached 53.51%. The compressed final packet averaged 527.5 bytes versus 1820.7 bytes for the original one-call arm (about 71% smaller), while all-call input and latency increased. Compression can preserve answer quality while still making the whole system more expensive.','',
'Claude completed after its quota reset. Original successful answers and the original format failure were retained; only requests with no model answer due to quota were resumed. The resume manifest records provenance. Generated accuracy improved from 2/6 to 5/6, including the format failure as zero. Natural F1 improved from 66.15% to 77.26%. These are small diagnostic samples, not reliable population gains.','',
'No claim is made that the model verifier proves correctness. It can miss unsupported claims or reject valid ones. The underlying graph remains an untyped co-activation graph. Source records are kept verbatim; no extracted relation is persisted as an observed fact.','',
'Local-model results: Qwen3 1.7B natural F1 increased from 44.85% to 73.61% with one invalid-source selection counted as failure, while generated exact accuracy fell from 2/6 to 1/6. Qwen3 4B natural F1 fell from 60.28% to 52.70%, while generated exact accuracy stayed at 1/6. There is no consistent small-model winner across the two tasks. The 1.7B artifact is the stronger natural-answer option in this tiny pilot; neither local controller matched the large models on generated reasoning chains.','',
'Post-pilot production hardening: selection schemas now restrict source IDs to the actual candidate IDs and bound array lengths. Failed plan output is retained in the stage audit. This addresses the invalid-ID failure structurally, but the original failure remains in the reported pilot and the complete benchmark has not been rerun under the new constraints.','',
'## Integration','',
'`User → your agent → context memory → selector / missing-fact searches → compact source packet → Astra, Claude, or local model → support check → answer`. The agent owns the loop. Memory is external to the model. Ordinary app chats are not automatically connected.','',
'```sh',
'.venv/bin/assoc-memory --backend codex --model gpt-6-astra --budget 8',
'.venv/bin/assoc-memory --backend claude --model claude-sonnet-4-6 --budget 8',
'```','',
'Use `/remember TEXT`, `/session`, and `/quit`. `--facts FILE.json` replaces the demo with a JSON list of observation strings. In-process RAM lasts for that CLI session. Generated answers are not silently remembered.','',
'Local transport uses a warmed loopback llama.cpp server; CLI backend latency includes CLI startup, so cross-backend latency is not a pure inference comparison. The local adapter uses structured JSON, temperature 0, seed 42, thinking disabled, and 1024 maximum output tokens. Server context is 8192. Qwen3 1.7B Q8_0 and Qwen3 4B Q4_K_M are compared; quantization differs, so the comparison is between these artifacts, not pure parameter count. Model downloads must pass the pinned official SHA-256 before use.','',
'## Live launcher check','',
'`tools/local_chat.sh 4 --budget 2 --question "Where does Mira keep her security token?"` verified the artifact, started a loopback server, ran the hardened pipeline, returned "In the blue drawer in the workshop", and stopped its own server. The pipeline reported 5.94 seconds, 3 calls, and 732 total input tokens; launcher startup and checksum time are additional. Original evidence JSON decreased from 505 to 161 bytes. The trace is `live-qwen4-launcher.json`. The launcher disables the optional RAM prompt cache to bound its extra cache residency; the earlier pilot used the server defaults.','',
'## Validation and reproducibility','',
'46 Python tests and 2 native CTests passed, including preserving a bridge fact during second retrieval, invalid source rejection, support-check abstention, transport checks, and CSR stress under UBSan. TSan remains unavailable due to its runtime initialization failure.','',
'Locked cases, protocol, per-question attempts, retrieval stages, drafts, citations, and usage are in this folder. Failures are retained rather than resampled. Run `benchmarks/pipeline_report.py` then `benchmarks/pipeline_markdown.py` to regenerate the tables.','']
oracle_file=OUT/'oracle-summary.json'
if oracle_file.exists():
    lines+=['## Complete-facts diagnostics','',
            'These controls supply all generated task facts and do not test production retrieval.','']
    for label,r in json.loads(oracle_file.read_text()).items():
        lines.append(f"- {label}: {r['correct']}/{r['n']} exact answers; {r['errors']} errors.")
    lines+=['', 'The gap between production retrieval and complete-facts answering helps locate failures in selection, query planning, and final reasoning. It does not prove improved model intelligence.','']
(OUT/'REPORT.md').write_text('\n'.join(lines))
print(OUT/'REPORT.md')
