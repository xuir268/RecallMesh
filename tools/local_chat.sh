#!/bin/sh
# Start a verified local model, run memory chat, then stop only our own server.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
model_size=${1:-4}
if [ "$#" -gt 0 ]; then shift; fi
case "$model_size" in
  1.7) model_file=models/Qwen3-1.7B-Q8_0.gguf; model_alias=qwen3-1.7b; model_sha=061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a ;;
  4) model_file=models/Qwen3-4B-Q4_K_M.gguf; model_alias=qwen3-4b; model_sha=7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5 ;;
  *) echo 'Usage: tools/local_chat.sh [1.7|4] [assoc-memory options]' >&2; exit 2 ;;
esac
.venv/bin/python - "$model_file" "$model_sha" <<'PY'
import hashlib,sys
from pathlib import Path
with Path(sys.argv[1]).open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
if actual!=sys.argv[2]:raise SystemExit('Model checksum mismatch; complete the verified download first.')
PY
server_log=$(mktemp /tmp/assoc-local-server.XXXXXX)
build/llama-runtime/bin/llama-server -m "$model_file" --alias "$model_alias" \
  --host 127.0.0.1 --port 8088 -c 8192 -np 1 -ngl 99 --cache-ram 0 > "$server_log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT HUP INT TERM
.venv/bin/python - "$server_pid" "$server_log" <<'PY'
import os,sys,time
from urllib.request import urlopen
for _ in range(60):
    try:os.kill(int(sys.argv[1]),0)
    except ProcessLookupError:raise SystemExit('Local server failed; inspect '+sys.argv[2])
    try:
        with urlopen('http://127.0.0.1:8088/health',timeout=1) as response:
            if response.status==200:break
    except Exception:pass
    time.sleep(1)
else:raise SystemExit('Local server not ready; inspect '+sys.argv[2])
PY
.venv/bin/assoc-memory --backend local --model "$model_alias" --local-url http://127.0.0.1:8088 --budget 8 "$@"
