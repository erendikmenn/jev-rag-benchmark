from jev_rag_benchmark.io import select_shard


def test_select_shard_partitions_rows_without_overlap():
    rows = [{"query_id": str(index)} for index in range(11)]
    shards = [select_shard(rows, index, 3) for index in range(3)]

    assert sum((shard for shard in shards), []) != rows
    assert {row["query_id"] for shard in shards for row in shard} == {
        row["query_id"] for row in rows
    }
    assert sum(len(shard) for shard in shards) == len(rows)
