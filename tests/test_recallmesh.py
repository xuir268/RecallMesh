import json
import subprocess
import sys
from recallmesh import MemoryClient, ManagedMemory, MemoryConfig


def test_named_entrypoint_and_public_api(tmp_path):
 config=tmp_path/'memory.json'
 subprocess.run([sys.executable,'-m','recallmesh','init','--config',str(config)],check=True,capture_output=True)
 with MemoryClient(config) as client:
  nid=client.remember('Mira keeps the token in a blue drawer.')['id']
 with ManagedMemory(MemoryConfig.load(config)) as memory:
  assert memory.get(nid)['text'].endswith('blue drawer.')
