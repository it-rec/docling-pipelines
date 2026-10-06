"""
JobStatsStore Port - Interface for job statistics storage

Implemented adapters:
- InMemoryJobStatsStore (testing/development)
- JsonJobStatsStore (filesystem-based persistent storage)

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from docpipe.core.constants import ExecutionStatus

if TYPE_CHECKING:
    from docpipe.core.job_management.domain.models import JobStats, NodeStats


class JobStatsStore(ABC):
    """
    Port interface for job statistics storage.
    This abstraction allows pluggable storage backends:
    - PostgreSQLJobStatsStore: PostgreSQL with atomic operations
    - InMemoryJobStatsStore: In-memory for testing
    - JsonJobStatsStore: Filesystem-based storage for simple persistence

    Responsibilities:
    - Persist and retrieve job-level statistics
    - Persist and retrieve node-level statistics
    - Support bulk operations for micro-batching
    - Provide atomic operations for concurrent updates
    """

    @abstractmethod
    def store_job_stats(self, job_stats: JobStats) -> None:
        """
        Store or update job-level statistics (UPSERT).

        Implementations should create a new record if job_run_id does not exist,
        or update the existing record if it does.

        Args:
            job_stats: Job statistics to store

        Raises:
            JobStatsStoreWriteException: If storage operation fails
        """
        ...

    @abstractmethod
    def get_job_stats(self, job_run_id: str) -> JobStats | None:
        """
        Retrieve job-level statistics.

        Args:
            job_run_id: Job run identifier

        Returns:
            JobStats if found, None otherwise

        Raises:
            JobStatsStoreReadException: If read operation fails
        """
        ...

    @abstractmethod
    def store_node_stats(self, *, job_run_id: str, node_stats: NodeStats) -> None:
        """
        Store or update node-level statistics (UPSERT).

        Identifies records by (node_id, job_run_id, batch_id) combination.

        Args:
            job_run_id: Job run identifier (globally unique)
            node_stats: Node statistics to store

        Raises:
            JobStatsStoreWriteException: If storage operation fails
        """
        ...

    @abstractmethod
    def get_node_stats(self, *, job_run_id: str) -> list[NodeStats]:
        """
        Retrieve ALL node statistics records (NO AGGREGATION).

        Important: Just fetch raw data, no aggregation logic here!
        Returns a flat list of all node stats (batch and non-batch).
        Aggregation is handled by NodeStatsAggregator service layer.

        Args:
            job_run_id: Job run identifier (globally unique)

        Returns:
            List of ALL NodeStats records

        Raises:
            JobStatsStoreReadException: If read operation fails
        """
        ...

    @abstractmethod
    def get_batch_node_stats(self, *, job_run_id: str) -> dict[str, dict[str, NodeStats]]:
        """
        Retrieve batch-level node statistics for micro-batching.

        Returns nested dictionary grouped by node_id, then batch_id.
        Used for detailed batch-level progress tracking.

        Args:
            job_run_id: Job run identifier (globally unique)

        Returns:
            Nested dict: {node_id: {batch_id: NodeStats}}

        Raises:
            JobStatsStoreReadException: If read operation fails
        """
        ...

    @abstractmethod
    def get_failed_docs_for_batch(self, *, job_run_id: str, batch_id: str) -> list[str]:
        """
        Retrieve failed document IDs for all nodes in a single batch.

        Scoped to one batch_id — proportional to one batch, not the whole job.

        Args:
            job_run_id: Job run identifier
            batch_id: Batch identifier to scope the query

        Returns:
            Flat list of failed document IDs for that batch

        Raises:
            JobStatsStoreReadException: If read operation fails
        """
        ...

    def get_node_stats_for_node(self, *, job_run_id: str, node_id: str) -> list[NodeStats]:
        """
        Retrieve raw node statistics records (batch and non-batch) of a single node.

        Returns exactly the records of ``get_node_stats()`` whose ``id`` equals
        ``node_id``, in the same relative order. The default implementation
        filters the full read; backends should override it to read only the
        records of that node.

        Args:
            job_run_id: Job run identifier (globally unique)
            node_id: Node identifier

        Returns:
            List of NodeStats records of that node

        Raises:
            JobStatsStoreReadException: If read operation fails
        """
        return [record for record in self.get_node_stats(job_run_id=job_run_id) if record.id == node_id]

    def get_batch_node_stats_for_node(self, *, job_run_id: str, node_id: str) -> dict[str, NodeStats]:
        """
        Retrieve batch-level node statistics of a single node.

        Returns the same mapping as ``get_batch_node_stats()[node_id]`` (or an
        empty dict). The default implementation filters the full read; backends
        should override it to read only the records of that node.

        Args:
            job_run_id: Job run identifier (globally unique)
            node_id: Node identifier

        Returns:
            Dict: {batch_id: NodeStats}

        Raises:
            JobStatsStoreReadException: If read operation fails
        """
        return self.get_batch_node_stats(job_run_id=job_run_id).get(node_id, {})

    def get_node_stats_with_batch_view(
        self, *, job_run_id: str
    ) -> tuple[list[NodeStats], dict[str, dict[str, NodeStats]]]:
        """
        Retrieve the results of ``get_node_stats()`` and ``get_batch_node_stats()`` together.

        Backends that can serve both views from a single read should override
        this. The NodeStats objects of the batch view must not share mutable
        state with the records of the flat list, because aggregation mutates
        the metadata of the flat records in place.

        Args:
            job_run_id: Job run identifier (globally unique)

        Returns:
            Tuple of (flat list of all NodeStats records, {node_id: {batch_id: NodeStats}})

        Raises:
            JobStatsStoreReadException: If read operation fails
        """
        return self.get_node_stats(job_run_id=job_run_id), self.get_batch_node_stats(job_run_id=job_run_id)

    @abstractmethod
    def bulk_store_node_stats(self, *, job_run_id: str, node_stats_list: list[NodeStats]) -> None:
        """
        Bulk store multiple node statistics (micro-batching).

        Optimized for creating pending node stats for all batches at once.

        Args:
            job_run_id: Job run identifier (globally unique)
            node_stats_list: List of node statistics to store

        Raises:
            JobStatsStoreWriteException: If bulk operation fails
        """
        ...

    @abstractmethod
    def atomic_increment_fields(
        self,
        job_run_id: str,
        increments: dict[str, int],
        updates: dict[str, Any] | None = None,
        jsonb_merges: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        Atomically increment numeric fields and update others.

        Critical for concurrent updates in micro-batching.
        PostgreSQL implementations use SQL atomic operations for thread-safety.

        Args:
            job_run_id: Job run identifier
            increments: Fields to increment {field_name: increment_value}
            updates: Fields to update {field_name: new_value}
            jsonb_merges: JSONB fields to merge (PostgreSQL-specific)

        Raises:
            JobStatsStoreWriteException: If atomic update fails
        """
        ...

    @abstractmethod
    def get_node_stats_by_batch_and_node(
        self, job_run_id: str, node_id: str, batch_id: str | None = None
    ) -> NodeStats | None:
        """
        Get specific node stats for batch and node combination.

        Args:
            job_run_id: Job run identifier
            node_id: Node identifier
            batch_id: Batch identifier (None for aggregated stats)

        Returns:
            NodeStats if found, None otherwise

        Raises:
            JobStatsStoreReadException: If read operation fails
        """
        ...

    @abstractmethod
    def delete_job_stats(self, job_run_id: str) -> None:
        """
        Delete job statistics and all associated node statistics.

        Implementations should perform a cascading delete if supported.

        Args:
            job_run_id: Job run identifier

        Raises:
            JobStatsStoreDeleteException: If job_run_id not found or deletion fails
        """
        ...

    @abstractmethod
    def list_job_runs(
        self,
        job_id: str | None = None,
        job_ids: list[str] | None = None,
        status: ExecutionStatus | str | None = None,
        limit: int = 100,
    ) -> list[JobStats]:
        """
        List job runs with optional filters.

        Args:
            job_id: Optional filter by a single job_id
            job_ids: Optional filter by a set of job_ids (bulk lookup, uses IN clause in SQL stores)
            status: Optional filter by status
            limit: Maximum number of results

        Returns:
            List of JobStats matching filters (sorted by start_time desc)

        Raises:
            JobStatsStoreReadException: If list operation fails
        """
        ...
