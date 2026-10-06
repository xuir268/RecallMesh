"""Model-independent agent connector over the memory CLI's JSON stdio protocol."""
import json
from pathlib import Path
from queue import Queue,Empty
import subprocess
import sys
import threading


class MemoryClientError(RuntimeError):
    def __init__(self,error):
        self.code=error['code'];super().__init__(error['message'])


class MemoryClient:
    def __init__(self,config='memory.json',timeout=30,command=None):
        config=str(Path(config).resolve());self.timeout=timeout;self._lock=threading.Lock();self._next_id=0
        if command is not None and (isinstance(command,str) or not command):
            raise ValueError('command must be a nonempty argument sequence')
        base=list(command) if command is not None else [sys.executable,'-m','assoc_mem.cli']
        self._process=subprocess.Popen(base+['serve','--config',config],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL,text=True,encoding='utf-8')
        self._responses=Queue();self._closed=False
        def reader():
            for line in self._process.stdout:
                try:self._responses.put(json.loads(line))
                except ValueError:self._responses.put({'ok':False,'error':{'code':'invalid_response','message':'CLI did not return JSON'}})
            self._responses.put(None)
        self._reader=threading.Thread(target=reader,daemon=True);self._reader.start()

    def call(self,method,**params):
        with self._lock:
            if self._closed:raise RuntimeError('MemoryClient is closed')
            self._next_id+=1;request_id=self._next_id
            try:
                self._process.stdin.write(json.dumps({'id':request_id,'method':method,'params':params},ensure_ascii=False)+'\n')
                self._process.stdin.flush()
                result=self._responses.get(timeout=self.timeout)
            except Empty:
                self.close();raise TimeoutError('Memory CLI request timed out') from None
            except (BrokenPipeError,OSError):
                self.close();raise RuntimeError('Memory CLI exited') from None
            if result is None:self.close();raise RuntimeError('Memory CLI exited before replying')
            if not result.get('ok'):raise MemoryClientError(result.get('error',{'code':'invalid_response','message':'Invalid CLI response'}))
            if result.get('id')!=request_id:self.close();raise RuntimeError('Memory CLI response ID mismatch')
            return result['result']

    def remember(self,text,writer=''):return self.call('remember',text=text,writer=writer)
    def recall(self,query,limit=None):
        return self.call('recall',**({'query':query,'limit':limit} if limit is not None else {'query':query}))
    def forget(self,nid):return self.call('forget',id=nid)
    def checkpoint(self):return self.call('checkpoint')
    def stats(self):return self.call('stats')
    def advance(self,steps=1):return self.call('advance',steps=steps)
    def close(self):
        if self._closed:return
        self._closed=True
        try:self._process.stdin.close()
        except OSError:pass
        try:self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill();self._process.wait()
        self._reader.join(timeout=1)
        self._process.stdout.close()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
