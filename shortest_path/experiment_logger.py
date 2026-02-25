from typing import Any

from causal_discovery.experiment_logger import ExperimentLogger


class ShortestPathLogger(ExperimentLogger):
    """ExperimentLogger adapted for shortest path results."""

    def _coerce(self, record: dict[str, Any]) -> dict[str, Any]:
        """Format shortest path fields for CSV serialization."""
        record = dict(record)

        # Convert path list to comma-separated string for CSV
        if "path" in record and isinstance(record["path"], list):
            record["path"] = ",".join(str(n) for n in record["path"])

        # Ensure total_weight is int
        if "total_weight" in record and record["total_weight"] is not None:
            try:
                record["total_weight"] = int(record["total_weight"])
            except (ValueError, TypeError):
                pass

        return record
