from pathlib import Path
import pytest


def test_oracle_preserves_sources_and_equal_record_budget(monkeypatch):
    benchmarks=Path(__file__).resolve().parents[1]/'benchmarks'
    monkeypatch.syspath_prepend(str(benchmarks))
    from small_answer_eval import oracle_packet
    records={i:{'id':f'D1:{i}','text':f'original source {i}'} for i in range(1,21)}
    packet=oracle_packet(['D1:19','D1:20'],records,[records[1],records[19],records[2]])
    assert len(packet)==16
    assert len({r['id'] for r in packet})==16
    assert {'D1:19','D1:20'}<={r['id'] for r in packet}
    assert all(r in records.values() for r in packet)
    with pytest.raises(ValueError,match='budget'):
        oracle_packet([r['id'] for r in records.values()],records,[])
