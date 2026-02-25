import logging
from typing import Any

from tqdm import tqdm

from .stages import Stage
from causal_discovery.utils import chunk_list
from causal_discovery.experiment_logger import ExperimentLogger


class CausalDiscoveryPipeline:
    def __init__(self, stages: list[Stage], logger: ExperimentLogger):
        self.stages = stages
        self.logger = logger

    def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Run all stages sequentially on a single sample.
        """
        for stage in self.stages:
            input_data = stage.process(input_data)
        self.logger.append(input_data)
        return input_data

    def run_seq_batch(self, input_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Run all stages sequentially on a batch of samples.
        """
        for input_data in input_samples:
            input_data = self.run(input_data)
        return input_samples

    def run_batch(self, input_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Run all stages on a batch of samples.
        """
        for stage in self.stages:
            input_samples = stage.process_batch(input_samples)
        self.logger.append_many(input_samples)
        return input_samples


class BatchCasualDiscoveryPipeline:
    """
    Pipeline that extends CausalDiscoveryPipeline to handle batch processing.
    Failed samples flow through with None values and are collected at the end.
    Retries are handled by the OpenAI SDK (max_retries on the client).
    """
    def __init__(self, pipeline: CausalDiscoveryPipeline, batch_size: int = 4):
        """
        :param pipeline: Instance of the single-sample pipeline.
        :param batch_size: Number of samples per batch.
        """
        self.pipeline = pipeline
        self.batch_size = batch_size

    def run_batch(self, input_samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
        results = []
        failed_ids = []
        batches = chunk_list(input_samples, self.batch_size)
        logging.info(f"Processing {len(input_samples)} samples in {len(batches)} batches.")

        for batch in tqdm(batches, desc="Processing batches"):
            batch_ids = [sample["sample_id"] for sample in batch]
            print(f"\nProcessing batch with sample IDs: {batch_ids}")

            batch_results = self.pipeline.run_batch(batch)
            results.extend(batch_results)

        return results, failed_ids
