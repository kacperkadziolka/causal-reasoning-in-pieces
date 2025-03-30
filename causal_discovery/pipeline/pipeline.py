import logging
import time
from typing import Any

from tqdm import tqdm

from .stages import Stage
from causal_discovery.utils import chunk_list


class CausalDiscoveryPipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Run all stages sequentially on a single sample.
        """
        for stage in self.stages:
            input_data = stage.process(input_data)
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
        return input_samples


class BatchCasualDiscoveryPipeline:
    """
    Pipeline that extend the CausalDiscoveryPipeline to handle batch processing with retries.
    """
    def __init__(self, pipeline: CausalDiscoveryPipeline, batch_size: int = 4, max_retries: int = 3, retry_delay: float = 1.0):
        """
        :param batch_size: Batch size for retries.
        :param pipeline: Instance of the single-sample pipeline.
        :param max_retries: Maximum number of retries per batch.
        :param retry_delay: Delay (in seconds) between retries.
        """
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def run_batch(self, input_samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
        results = []
        failed_ids = []
        batches = chunk_list(input_samples, self.batch_size)
        logging.info(f"Processing {len(input_samples)} samples in {len(batches)} batches.")

        for batch in tqdm(batches, desc="Processing batches"):
            retries = 0
            success = False
            while retries < self.max_retries and not success:
                try:
                    batch_results = self.pipeline.run_batch(batch)
                    results.extend(batch_results)
                    success = True
                except Exception as e:
                    retries += 1
                    logging.warning(f"Batch failed on attempt {retries}: {e}")
                    time.sleep(self.retry_delay)
            if not success:
                batch_ids = [sample["sample_id"] for sample in batch]
                failed_ids.extend(batch_ids)
                logging.error(f"Max retries reached for batch with sample IDs: {batch_ids}. Skipping these samples.")
        return results, failed_ids
