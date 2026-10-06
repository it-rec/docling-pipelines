"""
NodeStatsAggregator - Centralized aggregation logic for node statistics.

This service encapsulates ALL aggregation logic so that JobStatsStore
implementations don't need to duplicate it. Stores only fetch raw data,
and this service handles the aggregation.
"""

from collections import defaultdict

from docpipe.core.constants.constants import DocpipeConstants, ExecutionStatus, Metrics
from docpipe.core.constants.operator_constants import OperatorConstants
from docpipe.core.job_management.application.aggregation import MetadataAggregator, aggregate_batch_node_stats
from docpipe.core.job_management.domain.models import NodeStats
from docpipe.core.job_management.domain.ports import JobStatsStore
from docpipe.utils.infrastructure.logging import get_logger

logger = get_logger()


class NodeStatsAggregator:
    """
    Common aggregation layer for node statistics.

    This class encapsulates ALL aggregation logic so that JobStatsStore
    implementations don't need to duplicate it. Stores only fetch raw data,
    and this class handles the aggregation.

    Benefits:
    - Single source of truth for aggregation logic
    - Stores remain simple (just data access)
    - Consistent behavior across all storage backends
    - Easy to test and maintain
    """

    def __init__(self, *, job_stats_store: JobStatsStore):
        """
        Initialize NodeStatsAggregator.

        Args:
            job_stats_store: Storage adapter for fetching raw node stats
        """
        self.job_stats_store = job_stats_store
        self.metadata_aggregator = MetadataAggregator()

    def get_aggregated_node_stats(self, *, job_id: str, job_run_id: str) -> dict[str, NodeStats]:
        """
        Get aggregated node statistics.

        This method:
        1. Fetches ALL raw node stats from store
        2. Separates batch vs non-batch records
        3. Groups batch records by node_id
        4. Applies shared aggregation logic
        5. Returns aggregated results

        Args:
            job_id: Job identifier
            job_run_id: Job run identifier

        Returns:
            Dictionary mapping node_id to aggregated NodeStats
        """
        all_records = self.job_stats_store.get_node_stats(job_run_id=job_run_id)
        return self.aggregate_node_stats_records(all_records=all_records)

    def get_aggregated_node_stats_for_node(self, *, job_run_id: str, node_id: str) -> NodeStats | None:
        """
        Get the aggregated statistics of a single node.

        Reads only the raw records of that node and applies the same aggregation
        as ``get_aggregated_node_stats()``, so the result equals
        ``get_aggregated_node_stats(...).get(node_id)``.

        Args:
            job_run_id: Job run identifier
            node_id: Node identifier

        Returns:
            Aggregated NodeStats, or None if the node has no reportable records
        """
        node_records = self.job_stats_store.get_node_stats_for_node(job_run_id=job_run_id, node_id=node_id)
        return self.aggregate_node_stats_records(all_records=node_records).get(node_id)

    def get_batch_node_stats_for_node(self, *, job_run_id: str, node_id: str) -> dict[str, NodeStats]:
        """
        Get batch-level node statistics of a single node (no aggregation).

        Args:
            job_run_id: Job run identifier
            node_id: Node identifier

        Returns:
            Dict: {batch_id: NodeStats}
        """
        return self.job_stats_store.get_batch_node_stats_for_node(job_run_id=job_run_id, node_id=node_id)

    def get_aggregated_and_batch_node_stats(
        self, *, job_run_id: str
    ) -> tuple[dict[str, NodeStats], dict[str, dict[str, NodeStats]]]:
        """
        Get aggregated node statistics and batch-level node statistics from one store read.

        Equivalent to calling ``get_aggregated_node_stats()`` and
        ``get_batch_node_stats()``, but lets the store serve both views from a
        single scan.

        Args:
            job_run_id: Job run identifier

        Returns:
            Tuple of ({node_id: aggregated NodeStats}, {node_id: {batch_id: NodeStats}})
        """
        all_records, batch_node_stats = self.job_stats_store.get_node_stats_with_batch_view(job_run_id=job_run_id)
        return self.aggregate_node_stats_records(all_records=all_records), batch_node_stats

    def aggregate_node_stats_records(self, *, all_records: dict | list) -> dict[str, NodeStats]:
        """
        Aggregate raw node statistics records into one NodeStats per node.

        Batch records are aggregated per node; a non-batch record of a node is
        returned as is (with progress percentage injected).

        Args:
            all_records: Raw records as returned by the store

        Returns:
            Dictionary mapping node_id to aggregated NodeStats
        """
        if not all_records:
            return {}

        records_list = self._normalise_records(all_records)
        if records_list is None:
            return {}

        batch_records = [r for r in records_list if getattr(r, DocpipeConstants.BATCH_ID, None) is not None]
        non_batch_records = [r for r in records_list if getattr(r, DocpipeConstants.BATCH_ID, None) is None]

        result = self._aggregate_batch_records(batch_records)

        for record in non_batch_records:
            node_id = getattr(record, "id", getattr(record, DocpipeConstants.NODE_ID, None))
            if node_id:
                self._inject_progress_percentage(record)
                result[node_id] = record

        return result

    def _normalise_records(self, all_records: dict | list) -> list[NodeStats] | None:
        """Coerce the raw store return value to a flat list of NodeStats records.

        Returns None when the type is unrecognised (and logs an error).
        """
        if isinstance(all_records, dict):
            return list(all_records.values())
        if isinstance(all_records, list):
            return all_records
        logger.error("Unexpected return type from get_node_stats: %s", type(all_records))
        return None

    def _aggregate_batch_records(self, batch_records: list[NodeStats]) -> dict[str, NodeStats]:
        """Group batch records by node and return one aggregated NodeStats per node.

        Nodes where every batch is still PENDING or QUEUED are skipped.
        """
        if not batch_records:
            return {}

        batch_records_by_node: dict[str, list[NodeStats]] = defaultdict(list)
        for record in batch_records:
            node_id = getattr(record, "id", getattr(record, DocpipeConstants.NODE_ID, None))
            if node_id:
                batch_records_by_node[node_id].append(record)

        result: dict[str, NodeStats] = {}
        for node_id, node_batch_records in batch_records_by_node.items():
            all_pending = all(
                getattr(r, "node_status", ExecutionStatus.PENDING.value)
                in (ExecutionStatus.PENDING.value, ExecutionStatus.QUEUED.value)
                for r in node_batch_records
            )
            if all_pending:
                continue

            aggregated_stats = aggregate_batch_node_stats(
                node_id=node_id, batch_records=node_batch_records, aggregator=self.metadata_aggregator
            )
            if aggregated_stats:
                result[node_id] = aggregated_stats

        return result

    @staticmethod
    def _inject_progress_percentage(record: NodeStats) -> None:
        """Inject progress_percentage (float 0-100) into node_metadata for non-batch records.

        Uses total_docs, docs_completed, failed_docs, skipped_docs (lists of doc IDs).
        Falls back to integer counts stored in node_metadata when the list is empty
        (operators that track counts rather than individual doc IDs).

        Same formula as the microbatching path (finished_batches / total_batches * 100).
        Mutates the record in-place before it is returned to the caller.
        """
        node_metadata = getattr(record, "node_metadata", None)
        if not isinstance(node_metadata, dict):
            return

        inner = node_metadata.get(OperatorConstants.Metadata.NODE_METADATA)
        if not isinstance(inner, dict):
            return

        # Derive total from the list of doc IDs; fall back to the integer count
        # stored inside node_metadata for operators that track counts, not IDs.
        total_list = getattr(record, "total_docs", None) or []
        total = len(total_list)
        if total == 0:
            total = inner.get(Metrics.External.TOTAL_DOCS, 0) or 0

        if total == 0:
            return

        processed = len(
            set(getattr(record, "docs_completed", None) or [])
            | set(getattr(record, "failed_docs", None) or [])
            | set(getattr(record, "skipped_docs", None) or [])
        )
        # If doc-ID lists are all empty but we have a count-based total, fall back
        # to processed_docs integer from node_metadata.
        if processed == 0 and total_list == []:
            processed = inner.get("processed_docs", 0) or 0

        pct = round((processed / total) * 100, 2)
        inner[OperatorConstants.Metadata.PROGRESS_PERCENTAGE] = pct

    def get_batch_node_stats(self, *, job_id: str, job_run_id: str) -> dict[str, dict[str, NodeStats]]:
        """
        Get batch-level node statistics (no aggregation).

        Returns raw batch records grouped by node_id, then batch_id.
        Matches micro-batching specification.

        Args:
            job_id: Job identifier
            job_run_id: Job run identifier

        Returns:
            Nested dict: {node_id: {batch_id: NodeStats}}
        """
        # Use store's get_batch_node_stats which already groups correctly
        return self.job_stats_store.get_batch_node_stats(job_run_id=job_run_id)
