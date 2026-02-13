from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class BaseTask(ABC):
    """Abstract base class for task configurations.

    Each task encapsulates algorithm name, description, data loading,
    result extraction, and evaluation logic for a specific problem type.
    """

    algorithm_name: str
    task_description: str
    default_dataset_path: str

    @abstractmethod
    def load_dataset(self, path: str) -> pd.DataFrame:
        """Load the dataset from the given path into a DataFrame."""
        ...

    @abstractmethod
    def fetch_sample(self, dataset: pd.DataFrame, sample_idx: Optional[int] = None) -> pd.Series:
        """Fetch a specific or random sample from the dataset."""
        ...

    @abstractmethod
    def extract_result(
        self,
        final_result: Any,
        final_key: str,
        plan: Any,
    ) -> Dict[str, Any]:
        """Extract task-specific result from the pipeline's final output.

        Returns a dict with extracted values (task-specific keys).
        Raises ValueError if extraction fails.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        extracted: Dict[str, Any],
        sample: pd.Series,
    ) -> Dict[str, Any]:
        """Evaluate extracted result against the ground truth sample.

        Returns a dict that MUST include at minimum:
            - "is_correct": bool
            - "predicted_summary": str  (short description of prediction)
            - "expected_summary": str   (short description of expected)
        Additional task-specific keys are allowed.
        """
        ...

    @abstractmethod
    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate evaluation results into summary metrics."""
        ...

    @abstractmethod
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print aggregated metrics in a human-readable format."""
        ...
