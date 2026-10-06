"""
Equivalence tests for the scoped reads of JobTrackerService / NodeStatsAggregator.

- get_job(include_node_stats=True, include_batch_stats=True) reads the node-stats
  records once and must return the same JobStats (and the same API response) as
  the previous two-read implementation.
- The per-node getters must return the matching slice of the full get_job views.
"""

from unittest.mock import patch

import pytest

from docpipe.api.dto.mappers.job_stats_mapper import JobStatsMapper
from docpipe.core.constants.constants import ExecutionStatus
from docpipe.core.job_management.adapters.services.job_tracker_service import JobTrackerService
from docpipe.core.job_management.adapters.stores.duckdb.duckdb_job_stats_store import DuckDBJobStatsStore
from docpipe.core.job_management.adapters.stores.inmemory.inmemory_job_stats_store import InMemoryJobStatsStore
from docpipe.core.job_management.adapters.stores.json.json_job_stats_store import JsonJobStatsStore
from docpipe.core.job_management.application.services import NodeStatsAggregator
from docpipe.core.job_management.domain.models import JobStats, NodeStats
from docpipe.core.job_management.domain.ports import JobStatsService

# UUID-shaped identifiers: the API DTOs validate id lengths.
JOB_ID = "a1111111-0000-0000-0000-000000000001"
JOB_RUN_ID = "b2222222-0000-0000-0000-000000000002"
INGEST_NODE = "c3333333-0000-0000-0000-000000000003"
EXTRACT_NODE = "d4444444-0000-0000-0000-000000000004"
CHUNK_NODE = "e5555555-0000-0000-0000-000000000005"
BATCH_IDS = [f"f666666{num}-0000-0000-0000-00000000000{num}" for num in range(3)]
ALL_NODES = [INGEST_NODE, EXTRACT_NODE, CHUNK_NODE, "f7777777-0000-0000-0000-000000000007"]


def _populate(*, service: JobTrackerService) -> None:
    """Drive the service through a partially finished micro-batched run."""
    service.start_tracking_job(job_id=JOB_ID, job_run_id=JOB_RUN_ID, flow_name="flow")
    ingest_docs = [f"doc-{i}" for i in range(9)]
    service.start_node_execution(job_run_id=JOB_RUN_ID, node_id=INGEST_NODE, node_name="ingest", total_docs=ingest_docs)
    service.complete_node_execution(
        job_run_id=JOB_RUN_ID,
        node_id=INGEST_NODE,
        node_name="ingest",
        docs_completed=ingest_docs,
        failed_docs=[],
        skipped_docs=[],
        col_names=["id", "content"],
        node_status=ExecutionStatus.COMPLETED.value,
        node_metadata={"processed_docs": 9},
    )
    service.create_pending_batch_node_stats(
        job_run_id=JOB_RUN_ID,
        batch_ids=BATCH_IDS,
        batch_nums=list(range(len(BATCH_IDS))),
        downstream_node_ids=[EXTRACT_NODE, CHUNK_NODE],
        downstream_node_names=["extract", "chunk"],
    )
    for num, batch_id in enumerate(BATCH_IDS[:2]):
        docs = ingest_docs[num * 3 : num * 3 + 3]
        for node_id, name in ((EXTRACT_NODE, "extract"), (CHUNK_NODE, "chunk")):
            service.start_node_execution(
                job_run_id=JOB_RUN_ID,
                node_id=node_id,
                node_name=name,
                total_docs=docs,
                batch_id=batch_id,
                batch_num=num,
            )
            service.complete_node_execution(
                job_run_id=JOB_RUN_ID,
                node_id=node_id,
                node_name=name,
                docs_completed=docs[:2],
                failed_docs=docs[2:],
                skipped_docs=[],
                col_names=["id", "content"],
                node_status=ExecutionStatus.COMPLETED_WITH_ERRORS.value,
                node_metadata={"processed_docs": 3, "failed_docs": 1},
                batch_id=batch_id,
                batch_num=num,
            )
    # Batch 2 of the extract node is running and carries transient extraction progress,
    # which aggregation pops from the record metadata in place.
    service.job_stats_store.store_node_stats(
        job_run_id=JOB_RUN_ID,
        node_stats=NodeStats(
            id=EXTRACT_NODE,
            name="extract",
            node_status=ExecutionStatus.RUNNING.value,
            batch_id=BATCH_IDS[2],
            batch_num=2,
            total_docs=ingest_docs[6:],
            node_metadata={
                "id": EXTRACT_NODE,
                "operator": "extract",
                "node_metadata": {
                    "extraction_running": 3,
                    "extraction_completed": 1,
                    "progress_percentage": 33.3,
                    "extraction_stage_progress": {
                        "parse": {"documents_total": 3, "documents_completed": 1, "status": "Running"}
                    },
                },
            },
        ),
    )


@pytest.fixture(params=["json", "duckdb", "inmemory"])
def service(request, tmp_path) -> JobTrackerService:
    if request.param == "json":
        store = JsonJobStatsStore(base_dir=tmp_path / "job_stats", lock_timeout=5.0)
    elif request.param == "duckdb":
        store = DuckDBJobStatsStore(config={"database_path": str(tmp_path / "job_stats.duckdb")})
    else:
        store = InMemoryJobStatsStore()
    svc = JobTrackerService(job_stats_store=store, node_stats_aggregator=NodeStatsAggregator(job_stats_store=store))
    _populate(service=svc)
    return svc


def _legacy_get_job_with_batch_stats(*, service: JobTrackerService) -> JobStats:
    """The previous get_job(include_node_stats=True, include_batch_stats=True): two separate full reads."""
    job_stats = service.job_stats_store.get_job_stats(JOB_RUN_ID)
    assert job_stats is not None
    job_stats.node_stats = service.node_stats_aggregator.get_aggregated_node_stats(job_id=JOB_ID, job_run_id=JOB_RUN_ID)
    job_stats.batch_node_stats = service.node_stats_aggregator.get_batch_node_stats(
        job_id=JOB_ID, job_run_id=JOB_RUN_ID
    )
    return job_stats


class TestGetJobSingleScan:
    """get_job with node and batch stats must be unchanged while reading the records once."""

    def test_job_stats_identical_to_two_reads(self, *, service):
        expected = _legacy_get_job_with_batch_stats(service=service)

        result = service.get_job(job_run_id=JOB_RUN_ID, include_node_stats=True, include_batch_stats=True)

        assert result is not None
        assert result.model_dump_json() == expected.model_dump_json()

    @pytest.mark.parametrize("include_logs", [False, True])
    def test_api_status_response_byte_identical(self, *, service, include_logs):
        # The mapper derives a running job's duration from the wall clock; pin end_time so
        # both responses are computed from stored data only.
        job_stats = service.job_stats_store.get_job_stats(JOB_RUN_ID)
        job_stats.end_time = job_stats.start_time + 5
        service.job_stats_store.store_job_stats(job_stats)

        expected = JobStatsMapper.to_status_response(
            job_stats=_legacy_get_job_with_batch_stats(service=service), include_logs=include_logs
        )

        result = service.get_formatted_job_stats(job_run_id=JOB_RUN_ID, include_logs=include_logs)

        assert result.model_dump_json() == expected.model_dump_json()

    def test_transient_fields_survive_in_batch_view(self, *, service):
        """Aggregation pops transient metadata from the flat records only, never from the batch view."""
        result = service.get_job(job_run_id=JOB_RUN_ID, include_node_stats=True, include_batch_stats=True)

        assert result is not None
        running = result.batch_node_stats[EXTRACT_NODE][BATCH_IDS[2]]
        assert "extraction_running" in running.node_metadata["node_metadata"]
        assert "extraction_running" not in result.node_stats[EXTRACT_NODE].node_metadata.get("node_metadata", {})

    def test_reads_node_stats_once(self, *, service):
        store = service.job_stats_store
        with (
            patch.object(store, "get_node_stats", wraps=store.get_node_stats) as full_read,
            patch.object(store, "get_batch_node_stats", wraps=store.get_batch_node_stats) as batch_read,
        ):
            service.get_job(job_run_id=JOB_RUN_ID, include_node_stats=True, include_batch_stats=True)

        if isinstance(store, JsonJobStatsStore):
            full_read.assert_not_called()
            batch_read.assert_not_called()
        else:
            # Backends without a combined read fall back to the two separate reads.
            assert full_read.call_count == 1
            assert batch_read.call_count == 1

    def test_single_flag_paths_unchanged(self, *, service):
        only_nodes = service.get_job(job_run_id=JOB_RUN_ID, include_node_stats=True, include_batch_stats=False)
        only_batches = service.get_job(job_run_id=JOB_RUN_ID, include_node_stats=False, include_batch_stats=True)
        legacy = _legacy_get_job_with_batch_stats(service=service)

        assert only_nodes is not None and only_batches is not None
        assert {k: v.model_dump() for k, v in only_nodes.node_stats.items()} == {
            k: v.model_dump() for k, v in legacy.node_stats.items()
        }
        assert only_nodes.batch_node_stats == {}
        assert only_batches.node_stats == {}
        assert only_batches.model_dump()["batch_node_stats"] == legacy.model_dump()["batch_node_stats"]


class TestPerNodeGetters:
    """Per-node getters must return the matching slice of the full views."""

    @pytest.mark.parametrize("node_id", ALL_NODES)
    def test_aggregated_node_stats_for_node(self, *, service, node_id):
        full = service.get_job(job_run_id=JOB_RUN_ID, include_node_stats=True)
        assert full is not None
        expected = full.node_stats.get(node_id)

        result = service.get_aggregated_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=node_id)

        assert (result.model_dump() if result else None) == (expected.model_dump() if expected else None)

    @pytest.mark.parametrize("node_id", ALL_NODES)
    def test_batch_node_stats_for_node(self, *, service, node_id):
        full = service.get_job(job_run_id=JOB_RUN_ID, include_node_stats=False, include_batch_stats=True)
        assert full is not None
        expected = full.batch_node_stats.get(node_id, {})

        result = service.get_batch_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=node_id)

        assert [(k, v.model_dump()) for k, v in result.items()] == [(k, v.model_dump()) for k, v in expected.items()]

    @pytest.mark.parametrize("node_id", ALL_NODES)
    def test_port_defaults_agree_with_overrides(self, *, service, node_id):
        """The JobStatsService port fallbacks (based on get_job) give the same answers."""
        default_agg = JobStatsService.get_aggregated_node_stats_for_node(
            service, job_run_id=JOB_RUN_ID, node_id=node_id
        )
        default_batches = JobStatsService.get_batch_node_stats_for_node(service, job_run_id=JOB_RUN_ID, node_id=node_id)
        override_agg = service.get_aggregated_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=node_id)
        override_batches = service.get_batch_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=node_id)

        assert (default_agg.model_dump() if default_agg else None) == (
            override_agg.model_dump() if override_agg else None
        )
        assert {k: v.model_dump() for k, v in default_batches.items()} == {
            k: v.model_dump() for k, v in override_batches.items()
        }

    @pytest.mark.parametrize("node_id", ALL_NODES)
    def test_all_node_batches_in_statuses_matches_port_default(self, *, service, node_id):
        from docpipe.core.job_management.application.aggregation.batch_aggregator import FINISHED_BATCH_STATUSES

        for statuses in (FINISHED_BATCH_STATUSES, FINISHED_BATCH_STATUSES | {ExecutionStatus.RUNNING.value}):
            assert service.all_node_batches_in_statuses(
                job_run_id=JOB_RUN_ID, node_id=node_id, statuses=statuses
            ) == JobStatsService.all_node_batches_in_statuses(
                service, job_run_id=JOB_RUN_ID, node_id=node_id, statuses=statuses
            )

    def test_aggregated_node_stats_for_node_none_for_unknown_run(self, *, service):
        assert service.get_aggregated_node_stats_for_node(job_run_id="missing-run", node_id=EXTRACT_NODE) is None
        assert (
            JobStatsService.get_aggregated_node_stats_for_node(service, job_run_id="missing-run", node_id=EXTRACT_NODE)
            is None
        )
        assert JobStatsService.get_batch_node_stats_for_node(service, job_run_id="missing-run", node_id="x") == {}

    def test_per_node_getters_do_not_read_all_records(self, *, service):
        store = service.job_stats_store
        with (
            patch.object(store, "get_node_stats", wraps=store.get_node_stats) as full_read,
            patch.object(store, "get_batch_node_stats", wraps=store.get_batch_node_stats) as batch_read,
        ):
            service.get_aggregated_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=CHUNK_NODE)
            service.get_batch_node_stats_for_node(job_run_id=JOB_RUN_ID, node_id=CHUNK_NODE)

        full_read.assert_not_called()
        batch_read.assert_not_called()


class TestFailedDocIdsForBatch:
    """The service delegates to the batch-scoped store query; result equals filtering every record."""

    @pytest.mark.parametrize("batch_id", [*BATCH_IDS, "no-such-batch"])
    def test_matches_full_scan(self, *, service, batch_id):
        expected: list[str] = []
        for record in service.job_stats_store.get_node_stats(job_run_id=JOB_RUN_ID):
            if record.batch_id == batch_id and record.failed_docs:
                expected.extend(record.failed_docs)

        assert service.get_failed_doc_ids_for_batch(job_run_id=JOB_RUN_ID, batch_id=batch_id) == expected
