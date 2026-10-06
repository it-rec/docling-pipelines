"""
Equivalence tests for the scoped node-stats reads of the job stats stores.

The scoped reads (one node, one batch, or both node views from one scan) must
return exactly what the corresponding full read returned before, including
record order, for every store backend.
"""

from unittest.mock import patch

import pytest

from docpipe.core.constants.constants import ExecutionStatus
from docpipe.core.job_management.adapters.stores.duckdb.duckdb_job_stats_store import DuckDBJobStatsStore
from docpipe.core.job_management.adapters.stores.inmemory.inmemory_job_stats_store import InMemoryJobStatsStore
from docpipe.core.job_management.adapters.stores.json.json_job_stats_store import JsonJobStatsStore
from docpipe.core.job_management.domain.models import JobStats, NodeStats
from docpipe.core.job_management.domain.ports import JobStatsStore

JOB_RUN_ID = "run-scoped-reads"
INGEST_NODE = "ingest-node"
NODE_A = "node-a"
# Shares the "node-a_" file-name prefix with NODE_A's batch files in the JSON store.
NODE_A_PREFIXED = "node-a_extra"
NODE_B = "node-b"
BATCH_IDS = ["batch-0", "batch-1", "batch-2"]
ALL_NODES = [INGEST_NODE, NODE_A, NODE_A_PREFIXED, NODE_B, "unknown-node"]


def _populate(*, store: JobStatsStore) -> None:
    """Store a realistic mix of non-batch, pending, completed, failed and running records."""
    store.store_job_stats(JobStats(job_id="job", job_run_id=JOB_RUN_ID))
    store.store_node_stats(
        job_run_id=JOB_RUN_ID,
        node_stats=NodeStats(
            id=INGEST_NODE,
            name="ingest",
            node_status=ExecutionStatus.COMPLETED.value,
            total_docs=["d1", "d2"],
            docs_completed=["d1", "d2"],
            node_metadata={"id": INGEST_NODE, "operator": "ingest", "node_metadata": {"total_docs": 2}},
        ),
    )
    store.bulk_store_node_stats(
        job_run_id=JOB_RUN_ID,
        node_stats_list=[
            NodeStats(
                id=node_id, name=node_id, node_status=ExecutionStatus.PENDING.value, batch_id=batch_id, batch_num=num
            )
            for num, batch_id in enumerate(BATCH_IDS)
            for node_id in (NODE_A, NODE_A_PREFIXED, NODE_B)
        ],
    )
    for num, batch_id in enumerate(BATCH_IDS[:2]):
        docs = [f"doc-{num}-{k}" for k in range(4)]
        store.store_node_stats(
            job_run_id=JOB_RUN_ID,
            node_stats=NodeStats(
                id=NODE_A,
                name="a",
                node_status=ExecutionStatus.COMPLETED_WITH_ERRORS.value,
                batch_id=batch_id,
                batch_num=num,
                total_docs=docs,
                docs_completed=docs[:3],
                failed_docs=docs[3:],
                node_metadata={"id": NODE_A, "operator": "a", "node_metadata": {"processed_docs": 4}},
            ),
        )
        store.store_node_stats(
            job_run_id=JOB_RUN_ID,
            node_stats=NodeStats(
                id=NODE_B,
                name="b",
                node_status=ExecutionStatus.FAILED.value,
                batch_id=batch_id,
                batch_num=num,
                total_docs=docs,
                failed_docs=[f"b-fail-{num}", f"b-fail-{num}-2"],
                error=f"boom {num}",
            ),
        )
    # Running extraction-style record with transient fields that aggregation pops in place.
    store.store_node_stats(
        job_run_id=JOB_RUN_ID,
        node_stats=NodeStats(
            id=NODE_A_PREFIXED,
            name="x",
            node_status=ExecutionStatus.RUNNING.value,
            batch_id=BATCH_IDS[0],
            batch_num=0,
            total_docs=["doc-0-0"],
            failed_docs=["x-fail"],
            node_metadata={
                "id": NODE_A_PREFIXED,
                "operator": "x",
                "node_metadata": {"extraction_running": 2, "extraction_completed": 1, "progress_percentage": 50.0},
            },
        ),
    )


@pytest.fixture(params=["json", "duckdb", "inmemory"])
def populated_store(request, tmp_path) -> JobStatsStore:
    """Every store backend that runs without an external server, pre-populated."""
    if request.param == "json":
        store: JobStatsStore = JsonJobStatsStore(base_dir=tmp_path / "job_stats", lock_timeout=5.0)
    elif request.param == "duckdb":
        store = DuckDBJobStatsStore(config={"database_path": str(tmp_path / "job_stats.duckdb")})
    else:
        store = InMemoryJobStatsStore()
    _populate(store=store)
    return store


@pytest.fixture
def json_store(tmp_path) -> JsonJobStatsStore:
    """Pre-populated JSON store."""
    store = JsonJobStatsStore(base_dir=tmp_path / "job_stats", lock_timeout=5.0)
    _populate(store=store)
    return store


def _dumps(records: list[NodeStats]) -> list[dict]:
    return [record.model_dump() for record in records]


def _nested_dumps(view: dict[str, dict[str, NodeStats]]) -> list:
    """Dump a {node_id: {batch_id: NodeStats}} view preserving key order."""
    return [(node_id, [(bid, rec.model_dump()) for bid, rec in batches.items()]) for node_id, batches in view.items()]


class TestFailedDocsForBatch:
    """get_failed_docs_for_batch must equal filtering the full read by batch_id."""

    @pytest.mark.parametrize("batch_id", [*BATCH_IDS, "no-such-batch"])
    def test_matches_full_scan(self, *, populated_store, batch_id):
        expected: list[str] = []
        for record in populated_store.get_node_stats(job_run_id=JOB_RUN_ID):
            if record.batch_id == batch_id and record.failed_docs:
                expected.extend(record.failed_docs)

        result = populated_store.get_failed_docs_for_batch(job_run_id=JOB_RUN_ID, batch_id=batch_id)

        assert result == expected

    def test_batch_with_failures_is_not_empty(self, *, populated_store):
        result = populated_store.get_failed_docs_for_batch(job_run_id=JOB_RUN_ID, batch_id=BATCH_IDS[0])
        assert sorted(result) == sorted(["doc-0-3", "b-fail-0", "b-fail-0-2", "x-fail"])

    def test_json_reads_only_files_of_the_batch(self, *, json_store):
        with patch.object(json_store, "_read_json", wraps=json_store._read_json) as read_json:
            json_store.get_failed_docs_for_batch(job_run_id=JOB_RUN_ID, batch_id=BATCH_IDS[1])

        read_names = [call.kwargs["path"].name for call in read_json.call_args_list]
        assert read_names
        assert all(name.endswith(f"_{BATCH_IDS[1]}.json") for name in read_names)
        assert len(read_names) == 3  # one file per batch-participating node

    def test_duckdb_filters_in_sql(self, tmp_path):
        store = DuckDBJobStatsStore(config={"database_path": str(tmp_path / "scoped.duckdb")})
        _populate(store=store)
        with patch.object(store, "get_node_stats") as full_read:
            store.get_failed_docs_for_batch(job_run_id=JOB_RUN_ID, batch_id=BATCH_IDS[0])
        full_read.assert_not_called()


class TestNodeScopedReads:
    """Per-node reads must equal the matching slice of the full reads."""

    @pytest.mark.parametrize("node_id", ALL_NODES)
    def test_node_stats_for_node_matches_full_scan(self, *, populated_store, node_id):
        expected = [r for r in populated_store.get_node_stats(job_run_id=JOB_RUN_ID) if r.id == node_id]

        result = populated_store.get_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=node_id)

        assert _dumps(result) == _dumps(expected)

    @pytest.mark.parametrize("node_id", ALL_NODES)
    def test_batch_node_stats_for_node_matches_full_scan(self, *, populated_store, node_id):
        expected = populated_store.get_batch_node_stats(job_run_id=JOB_RUN_ID).get(node_id, {})

        result = populated_store.get_batch_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=node_id)

        assert [(k, v.model_dump()) for k, v in result.items()] == [(k, v.model_dump()) for k, v in expected.items()]

    @pytest.mark.parametrize("node_id", ALL_NODES)
    def test_port_default_implementations_agree(self, *, populated_store, node_id):
        """The port's fallback (filter the full read) returns the same as the backend override."""
        default_records = JobStatsStore.get_node_stats_for_node(populated_store, job_run_id=JOB_RUN_ID, node_id=node_id)
        default_batches = JobStatsStore.get_batch_node_stats_for_node(
            populated_store, job_run_id=JOB_RUN_ID, node_id=node_id
        )

        assert _dumps(default_records) == _dumps(
            populated_store.get_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=node_id)
        )
        assert {k: v.model_dump() for k, v in default_batches.items()} == {
            k: v.model_dump()
            for k, v in populated_store.get_batch_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=node_id).items()
        }

    def test_json_reads_only_files_of_the_node(self, *, json_store):
        with patch.object(json_store, "_read_json", wraps=json_store._read_json) as read_json:
            json_store.get_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=NODE_B)

        read_names = sorted(call.kwargs["path"].name for call in read_json.call_args_list)
        assert read_names == sorted(f"{NODE_B}_{batch_id}.json" for batch_id in BATCH_IDS)

    def test_json_non_batch_node_reads_single_file(self, *, json_store):
        with patch.object(json_store, "_read_json", wraps=json_store._read_json) as read_json:
            records = json_store.get_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=INGEST_NODE)
            batches = json_store.get_batch_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=INGEST_NODE)

        assert [r.id for r in records] == [INGEST_NODE]
        assert batches == {}
        assert [call.kwargs["path"].name for call in read_json.call_args_list] == [f"{INGEST_NODE}.json"]

    def test_json_missing_run_returns_empty(self, *, json_store):
        assert json_store.get_node_stats_for_node(job_run_id="missing-run", node_id=NODE_A) == []
        assert json_store.get_batch_node_stats_for_node(job_run_id="missing-run", node_id=NODE_A) == {}
        assert json_store.get_failed_docs_for_batch(job_run_id="missing-run", batch_id=BATCH_IDS[0]) == []


class TestNodeStatsWithBatchView:
    """get_node_stats_with_batch_view must equal the two separate full reads."""

    def test_matches_separate_reads(self, *, populated_store):
        expected_records = populated_store.get_node_stats(job_run_id=JOB_RUN_ID)
        expected_view = populated_store.get_batch_node_stats(job_run_id=JOB_RUN_ID)

        records, view = populated_store.get_node_stats_with_batch_view(job_run_id=JOB_RUN_ID)

        assert _dumps(records) == _dumps(expected_records)
        assert _nested_dumps(view) == _nested_dumps(expected_view)

    def test_batch_view_is_independent_of_flat_records(self, *, populated_store):
        records, view = populated_store.get_node_stats_with_batch_view(job_run_id=JOB_RUN_ID)
        expected_view = _nested_dumps(view)

        # Mutate the flat records the way aggregation does (pop nested metadata, touch lists).
        for record in records:
            if isinstance(record.node_metadata, dict):
                nested = record.node_metadata.get("node_metadata")
                if isinstance(nested, dict):
                    nested.clear()
                record.node_metadata.clear()
            record.total_docs.append("mutated")
            record.failed_docs.append("mutated")

        assert _nested_dumps(view) == expected_view

    def test_json_reads_each_file_once(self, *, json_store):
        node_stats_dir = json_store._get_node_stats_dir(job_run_id=JOB_RUN_ID)
        file_count = len(list(node_stats_dir.glob("*.json")))

        with patch.object(json_store, "_read_json", wraps=json_store._read_json) as read_json:
            json_store.get_node_stats_with_batch_view(job_run_id=JOB_RUN_ID)

        assert read_json.call_count == file_count

    def test_json_missing_run_returns_empty(self, *, json_store):
        assert json_store.get_node_stats_with_batch_view(job_run_id="missing-run") == ([], {})


class TestJsonFileListing:
    """The scandir-based listing must select files in the same order as Path.glob("*.json")."""

    def test_unfiltered_listing_matches_glob_order(self, *, json_store):
        node_stats_dir = json_store._get_node_stats_dir(job_run_id=JOB_RUN_ID)
        (node_stats_dir / "leftover.tmp").write_text("{}", encoding="utf-8")

        listed = JsonJobStatsStore._list_node_stats_files(node_stats_dir=node_stats_dir, name_filter=None)

        assert listed == list(node_stats_dir.glob("*.json"))

    def test_batch_file_name_matches_glob_pattern(self):
        import fnmatch

        for name in ["a_b.json", "a.json", "_.json", "a_.json", "_b.json", "node-a_extra.json", "nounderscore.json"]:
            assert JsonJobStatsStore._is_batch_file_name(name) == fnmatch.fnmatchcase(name, "*_*.json")

    def test_unreadable_directory_yields_nothing(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        assert JsonJobStatsStore._list_node_stats_files(node_stats_dir=missing, name_filter=None) == []
