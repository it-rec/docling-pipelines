"""
Tests for the job-stats reads issued by FlowExecutionEventHandler on every node step.

- Framework status payloads are only built when a job run manager will receive
  them, and node statistics are only aggregated for managers that consume them.
- The CLI operator summary reads only the current node's records and prints
  exactly when (and what) the previous full-scan implementation printed.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from docpipe.core.constants.constants import DocpipeConstants, ExecutionStatus
from docpipe.core.job_management.adapters.frameworks.default_job_run_manager import DefaultJobRunManager
from docpipe.core.job_management.adapters.services.job_tracker_service import JobTrackerService
from docpipe.core.job_management.adapters.stores.json.json_job_stats_store import JsonJobStatsStore
from docpipe.core.job_management.application.aggregation.batch_aggregator import FINISHED_BATCH_STATUSES
from docpipe.core.job_management.application.services import NodeStatsAggregator
from docpipe.core.job_management.domain.models import JobStats, NodeStats
from docpipe.core.job_management.domain.ports import JobRunManager
from docpipe.core.operators.abstract_operator import OperatorCategory
from docpipe.core.orchestration.flow_execution_event_handler import FlowExecutionEventHandler

JOB_ID = "job-handler"
JOB_RUN_ID = "run-handler"
INGEST_NODE = "ingest-node"
NODE_A = "node-a"
NODE_B = "node-b"
BATCH_IDS = ["batch-0", "batch-1", "batch-2"]


class _ConsumingJobRunManager(JobRunManager):
    """Custom framework adapter that keeps the port default (consumes node stats)."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def create_job_run(self, *, job_id: str, job_config: dict[str, Any]) -> dict[str, Any]:
        return {}

    def get_job_run(self, *, job_id: str, job_run_id: str) -> dict[str, Any]:
        return {}

    def update_job_run_status(
        self, *, job_run_id: str, status: str, job_run_stats: dict[str, Any] | None = None
    ) -> None:
        self.updates.append({"status": status, "job_run_stats": job_run_stats})

    def cancel_job_run(self, *, job_run_id: str) -> None:
        return None

    def delete_job_run(self, *, job_run_id: str) -> None:
        return None


def _step(*, handler: FlowExecutionEventHandler, node_id: str, global_config: dict | None) -> None:
    handler.after_step_execution_complete(
        node_id=node_id,
        node_name=f"name-{node_id}",
        operator_category=OperatorCategory.Functional,
        operator="op",
        global_config=global_config,
        is_last_step=False,
        metadata={},
        start_time=0,
        tables=None,
    )


def _make_handler(
    *, job_stats_service: Any, job_run_manager: Any = None, execution_reporter: Any = None
) -> FlowExecutionEventHandler:
    handler = FlowExecutionEventHandler(
        job_stats_service=job_stats_service,
        job_run_manager=job_run_manager,
        execution_reporter=execution_reporter,
    )
    handler.job_id = JOB_ID
    handler.job_run_id = JOB_RUN_ID
    handler.common_log_arguments = {}
    return handler


class TestLazyFrameworkStatus:
    """Job stats for framework updates are only read when a job run manager exists."""

    def test_step_without_manager_or_reporter_does_not_read_job(self):
        service = MagicMock()
        handler = _make_handler(job_stats_service=service)

        with patch.object(handler, "_get_complete_job_stats", wraps=handler._get_complete_job_stats) as collect:
            _step(handler=handler, node_id=NODE_A, global_config={DocpipeConstants.BATCH_ID: BATCH_IDS[0]})

        collect.assert_not_called()
        service.get_job.assert_not_called()
        service.update_doc_counts.assert_called_once()

    def test_node_failure_without_manager_skips_status_payload(self):
        service = MagicMock()
        handler = _make_handler(job_stats_service=service)

        with patch.object(handler, "_get_complete_job_stats") as collect:
            handler.after_node_failure(node_id=NODE_A, node_name="a", global_config={}, e=RuntimeError("boom"))

        collect.assert_not_called()

    def test_flow_complete_without_manager_skips_status_payload(self):
        service = MagicMock()
        service.get_job.return_value = None
        handler = _make_handler(job_stats_service=service)

        with patch.object(handler, "_get_complete_job_stats") as collect:
            handler.after_flow_execution_complete(
                op_flow=[], present_job_status=ExecutionStatus.CANCELING, message=None
            )

        collect.assert_not_called()

    def test_default_manager_gets_job_level_stats_without_node_aggregation(self):
        service = MagicMock()
        job_stats = JobStats(job_id=JOB_ID, job_run_id=JOB_RUN_ID, processed_docs=7)
        service.get_job.return_value = job_stats
        manager = DefaultJobRunManager(job_stats_service=service)
        handler = _make_handler(job_stats_service=service, job_run_manager=manager)

        with patch.object(manager, "update_job_run_status") as update:
            _step(handler=handler, node_id=NODE_A, global_config={})

        service.get_job.assert_called_once_with(job_run_id=JOB_RUN_ID, include_node_stats=False)
        update.assert_called_once_with(
            job_run_id=JOB_RUN_ID, status=ExecutionStatus.RUNNING.value, job_run_stats=job_stats.model_dump()
        )

    def test_custom_manager_still_receives_node_stats(self):
        service = MagicMock()
        node = NodeStats(id=NODE_A, name="a", node_status=ExecutionStatus.COMPLETED.value)
        service.get_job.return_value = JobStats(job_id=JOB_ID, job_run_id=JOB_RUN_ID, node_stats={NODE_A: node})
        manager = _ConsumingJobRunManager()
        handler = _make_handler(job_stats_service=service, job_run_manager=manager)

        handler.after_node_failure(node_id=NODE_A, node_name="a", global_config={}, e=RuntimeError("boom"))

        assert JobRunManager.consumes_node_stats is True
        service.get_job.assert_any_call(job_run_id=JOB_RUN_ID, include_node_stats=True)
        assert manager.updates[-1]["status"] == ExecutionStatus.FAILED.value
        payload = manager.updates[-1]["job_run_stats"]
        assert payload["message"] == "boom"
        assert payload["node_stats"][NODE_A]["node_status"] == ExecutionStatus.COMPLETED.value

    def test_default_job_run_manager_declares_no_node_stats(self):
        assert DefaultJobRunManager.consumes_node_stats is False


class TestOperatorSummaryReads:
    """Batch-mode summary checks completion from the node's batch records before aggregating."""

    def test_unfinished_batches_skip_aggregation(self):
        service = MagicMock()
        service.all_node_batches_in_statuses.return_value = False
        reporter = MagicMock()
        handler = _make_handler(job_stats_service=service, execution_reporter=reporter)

        _step(handler=handler, node_id=NODE_A, global_config={DocpipeConstants.BATCH_ID: BATCH_IDS[0]})

        service.all_node_batches_in_statuses.assert_called_once_with(
            job_run_id=JOB_RUN_ID, node_id=NODE_A, statuses=FINISHED_BATCH_STATUSES
        )
        service.get_aggregated_node_stats_for_node.assert_not_called()
        service.get_batch_node_stats_for_node.assert_not_called()
        service.get_job.assert_not_called()
        reporter.print_operator_summary.assert_not_called()

    @pytest.mark.parametrize("node_status", [ExecutionStatus.RUNNING.value, ExecutionStatus.COMPLETED.value])
    def test_finished_batches_print_only_terminal_aggregate(self, node_status):
        service = MagicMock()
        service.all_node_batches_in_statuses.return_value = True
        node = NodeStats(id=NODE_A, name="a", node_status=node_status)
        service.get_aggregated_node_stats_for_node.return_value = node
        reporter = MagicMock()
        handler = _make_handler(job_stats_service=service, execution_reporter=reporter)

        _step(handler=handler, node_id=NODE_A, global_config={DocpipeConstants.BATCH_ID: BATCH_IDS[0]})

        assert reporter.print_operator_summary.called == (node_status == ExecutionStatus.COMPLETED.value)

    def test_finished_status_set_matches_batch_completion_rule(self):
        """all-in-FINISHED_BATCH_STATUSES is the same rule as _all_batches_finished for every status mix."""
        import itertools

        statuses = [status.value for status in ExecutionStatus]
        for size in range(3):
            for combo in itertools.product(statuses, repeat=size):
                records = [NodeStats(id=NODE_A, name="a", node_status=status) for status in combo]
                expected = FlowExecutionEventHandler._all_batches_finished(batch_records=records)
                assert (bool(records) and all(s in FINISHED_BATCH_STATUSES for s in combo)) == expected, combo

    def test_non_batch_mode_prints_aggregated_node(self):
        service = MagicMock()
        node = NodeStats(id=INGEST_NODE, name="ingest", node_status=ExecutionStatus.RUNNING.value)
        service.get_aggregated_node_stats_for_node.return_value = node
        reporter = MagicMock()
        handler = _make_handler(job_stats_service=service, execution_reporter=reporter)

        _step(handler=handler, node_id=INGEST_NODE, global_config={})

        service.get_batch_node_stats_for_node.assert_not_called()
        service.get_job.assert_not_called()
        reporter.print_operator_summary.assert_called_once_with(
            step_name=f"name-{INGEST_NODE}", node_stats=node, tables=None
        )

    def test_unknown_node_does_not_print(self):
        service = MagicMock()
        service.get_aggregated_node_stats_for_node.return_value = None
        reporter = MagicMock()
        handler = _make_handler(job_stats_service=service, execution_reporter=reporter)

        _step(handler=handler, node_id=INGEST_NODE, global_config=None)

        reporter.print_operator_summary.assert_not_called()


def _legacy_summary(*, service: JobTrackerService, node_id: str, global_config: dict | None) -> NodeStats | None:
    """The previous decision: full get_job read + _should_print_operator_summary."""
    job_stats = service.get_job(job_run_id=JOB_RUN_ID, include_node_stats=True, include_batch_stats=True)
    if not (job_stats and job_stats.node_stats and node_id in job_stats.node_stats):
        return None
    node_stats = job_stats.node_stats[node_id]
    should_print = FlowExecutionEventHandler._should_print_operator_summary(
        node_id=node_id, node_stats=node_stats, global_config=global_config, job_stats=job_stats
    )
    return node_stats if should_print else None


class TestOperatorSummaryEquivalence:
    """End to end with the real JSON store: same summaries printed at the same steps as before."""

    @pytest.fixture
    def service(self, tmp_path) -> JobTrackerService:
        store = JsonJobStatsStore(base_dir=tmp_path / "job_stats", lock_timeout=5.0)
        svc = JobTrackerService(job_stats_store=store, node_stats_aggregator=NodeStatsAggregator(job_stats_store=store))
        svc.start_tracking_job(job_id=JOB_ID, job_run_id=JOB_RUN_ID, flow_name="flow")
        return svc

    @staticmethod
    def _run_node(*, service: JobTrackerService, node_id: str, status: str, batch_num: int | None) -> None:
        batch_id = BATCH_IDS[batch_num] if batch_num is not None else None
        docs = [f"doc-{batch_num}-{i}" for i in range(3)]
        service.start_node_execution(
            job_run_id=JOB_RUN_ID,
            node_id=node_id,
            node_name=node_id,
            total_docs=docs,
            batch_id=batch_id,
            batch_num=batch_num,
        )
        failed = docs[2:] if status != ExecutionStatus.COMPLETED.value else []
        service.complete_node_execution(
            job_run_id=JOB_RUN_ID,
            node_id=node_id,
            node_name=node_id,
            docs_completed=[d for d in docs if d not in failed],
            failed_docs=failed,
            skipped_docs=[],
            col_names=["id"],
            node_status=status,
            node_metadata={"processed_docs": len(docs)},
            batch_id=batch_id,
            batch_num=batch_num,
        )

    def test_prints_match_full_scan_decision(self, *, service):
        reporter = MagicMock()
        handler = _make_handler(job_stats_service=service, execution_reporter=reporter)

        # Non-batch ingest step.
        self._run_node(service=service, node_id=INGEST_NODE, status=ExecutionStatus.COMPLETED.value, batch_num=None)
        service.create_pending_batch_node_stats(
            job_run_id=JOB_RUN_ID,
            batch_ids=BATCH_IDS,
            batch_nums=list(range(len(BATCH_IDS))),
            downstream_node_ids=[NODE_A, NODE_B],
            downstream_node_names=[NODE_A, NODE_B],
        )
        steps: list[tuple[str, str, int | None]] = [(INGEST_NODE, ExecutionStatus.COMPLETED.value, None)]
        # Interleaved, out-of-order batch completions, including failures.
        steps += [
            (NODE_A, ExecutionStatus.COMPLETED.value, 0),
            (NODE_B, ExecutionStatus.FAILED.value, 0),
            (NODE_A, ExecutionStatus.COMPLETED.value, 2),
            (NODE_B, ExecutionStatus.COMPLETED.value, 2),
            (NODE_A, ExecutionStatus.COMPLETED_WITH_ERRORS.value, 1),
            (NODE_B, ExecutionStatus.COMPLETED.value, 1),
        ]

        printed_steps = []
        for index, (node_id, status, batch_num) in enumerate(steps):
            if index > 0:
                self._run_node(service=service, node_id=node_id, status=status, batch_num=batch_num)
            global_config = (
                {}
                if batch_num is None
                else {
                    DocpipeConstants.ENABLE_MICRO_BATCHING: True,
                    DocpipeConstants.BATCH_ID: BATCH_IDS[batch_num],
                    DocpipeConstants.BATCH_NUM: batch_num,
                }
            )
            reporter.reset_mock()

            _step(handler=handler, node_id=node_id, global_config=global_config)

            expected = _legacy_summary(service=service, node_id=node_id, global_config=global_config)
            if expected is None:
                reporter.print_operator_summary.assert_not_called()
                continue
            reporter.print_operator_summary.assert_called_once()
            kwargs = reporter.print_operator_summary.call_args.kwargs
            assert kwargs["step_name"] == f"name-{node_id}"
            assert kwargs["tables"] is None
            assert kwargs["node_stats"].model_dump() == expected.model_dump()
            printed_steps.append(index)

        # Ingest prints immediately; each batched node prints once, after its last batch.
        assert printed_steps == [0, 5, 6]
