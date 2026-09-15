"""Resume the pinned official model with validated byte-range downloads."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import os
import json
import time
from urllib.request import Request, urlopen

import argparse
parser=argparse.ArgumentParser();parser.add_argument('--size',choices=['1.7','4'],default='4');args=parser.parse_args()
URL='https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf'
SIZE=2497280256
SHA='7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5'
p=Path('models/Qwen3-4B-Q4_K_M.gguf')
if args.size=='1.7':
    URL='https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/90862c4b9d2787eaed51d12237eafdfe7c5f6077/Qwen3-1.7B-Q8_0.gguf'
    SIZE=1834426016
    SHA='061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a'
    p=Path('models/Qwen3-1.7B-Q8_0.gguf')
start=p.stat().st_size if p.exists() else 0
# Initial curl output is a contiguous prefix. Progress marker retains that
# boundary if a parallel download is interrupted and rerun.
marker=p.with_suffix('.resume')
if marker.exists():start=int(marker.read_text())
else:marker.write_text(str(start))
progress=p.with_suffix('.ranges.json')
completed=set(json.loads(progress.read_text())) if progress.exists() else set()
fd=os.open(p,os.O_CREAT|os.O_RDWR)
chunks=[(i,min(i+32*1024*1024,SIZE)-1) for i in range(start,SIZE,32*1024*1024) if i not in completed]
def chunk(bounds):
    a,b=bounds
    req=Request(URL+'?part='+str(a),headers={'Range':f'bytes={a}-{b}'})
    for attempt in range(5):
        try:
            with urlopen(req,timeout=180) as r:
                if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {a}-{b}/{SIZE}':
                    raise RuntimeError('Server did not return the requested byte range')
                data=r.read()
            if len(data)!=b-a+1:raise RuntimeError('Truncated byte range')
            break
        except Exception:
            if attempt==4:raise
            time.sleep(2**attempt)
    offset=0
    while offset<len(data):offset+=os.pwrite(fd,data[offset:],a+offset)
    return a,len(data)
try:
    total=start+sum(min(i+32*1024*1024,SIZE)-i for i in completed)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(chunk,c) for c in chunks]):
            offset,amount=f.result();completed.add(offset);total+=amount
            os.fsync(fd)
            temporary=progress.with_suffix('.tmp');temporary.write_text(json.dumps(sorted(completed)));temporary.replace(progress)
            print(f'{total/SIZE:.1%}',flush=True)
finally:os.close(fd)
with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
if actual!=SHA:raise RuntimeError('SHA256 mismatch: '+actual)
marker.unlink()
if progress.exists():progress.unlink()
print('SHA256 verified',actual)
