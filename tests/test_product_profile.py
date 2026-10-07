import json
from assoc_mem.config import MemoryConfig, PRODUCT_CONFIG
from assoc_mem.memory_cli import main


def test_init_uses_adjacency_without_changing_legacy_defaults(tmp_path, capsys):
    path = tmp_path/'memory.json'
    main(['init', '--config', str(path)])
    assert json.loads(capsys.readouterr().out)['ok']
    config = MemoryConfig.load(path)
    assert config['graph']['edge_policy'] == 'adjacent'
    assert config['graph']['lambda'] == 0
    assert MemoryConfig()['graph']['edge_policy'] == 'star'
    assert MemoryConfig()['graph']['lambda'] == .05
    assert MemoryConfig(PRODUCT_CONFIG)['graph']['edge_policy'] == 'adjacent'
