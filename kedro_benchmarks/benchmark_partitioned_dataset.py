"""ASV benchmarks for ``PartitionedDataset.load()``."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
import pandas as pd
from kedro_datasets.partitions import PartitionedDataset
from moto import mock_aws

from kedro.io.core import DatasetError

PARTITION_COUNTS = (10, 1000)
FILESYSTEMS = ("local", "s3")
REPEAT_COUNT = 5
S3_BUCKET_NAME = "kedro-partitioned-dataset-benchmarks"
S3_PREFIX = "partitions"


class CachedPartitionedDataset(PartitionedDataset):
    """A benchmark-only variant that preserves the old cached load behavior."""

    def load(self):
        partitions = {}

        for partition_file_path in self._list_partitions():
            kwargs = self._dataset_config.copy()
            kwargs[self._filepath_arg] = self._join_protocol(partition_file_path)
            dataset = self._dataset_type(**kwargs)  # type: ignore[misc]
            partition_id = self._path_to_partition(partition_file_path)
            partitions[partition_id] = dataset.load

        if not partitions:
            raise DatasetError(f"No partitions found in '{self._path}'")

        return partitions


class TimePartitionedDatasetLoad:
    """Benchmark the load path for local and mocked S3 partitioned datasets."""

    params = (FILESYSTEMS, PARTITION_COUNTS)
    param_names = ("filesystem", "partition_count")

    def setup(self, filesystem: str, partition_count: int):
        self._tempdir = TemporaryDirectory()
        self._mock_aws = mock_aws()
        self._mock_aws.start()

        self.partition_count = partition_count
        self.partition_data = self._build_partition_data(partition_count)

        if filesystem == "local":
            self.dataset_path = self._prepare_local_dataset()
        else:
            self.dataset_path = self._prepare_s3_dataset()

        self.current_dataset = PartitionedDataset(
            path=self.dataset_path,
            dataset="pandas.CSVDataset",
            filename_suffix=".csv",
        )
        self.cached_dataset = CachedPartitionedDataset(
            path=self.dataset_path,
            dataset="pandas.CSVDataset",
            filename_suffix=".csv",
        )

    def teardown(self, _filesystem: str, _partition_count: int):
        self._mock_aws.stop()
        self._tempdir.cleanup()

    def _build_partition_data(self, partition_count: int) -> dict[str, pd.DataFrame]:
        return {
            f"partition_{index:05d}/data.csv": pd.DataFrame(
                {"partition": [index], "value": [index * 2]}
            )
            for index in range(partition_count)
        }

    def _prepare_local_dataset(self) -> str:
        base_path = Path(self._tempdir.name) / f"local_{self.partition_count}"
        for partition_path, dataframe in self.partition_data.items():
            file_path = base_path / partition_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(dataframe.to_csv(index=False), encoding="utf-8")
        return str(base_path)

    def _prepare_s3_dataset(self) -> str:
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=S3_BUCKET_NAME)

        prefix = f"{S3_PREFIX}/{self.partition_count}"
        for partition_path, dataframe in self.partition_data.items():
            client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=f"{prefix}/{partition_path}",
                Body=dataframe.to_csv(index=False).encode("utf-8"),
            )
        return f"s3://{S3_BUCKET_NAME}/{prefix}"

    def time_load_current(self, _filesystem: str, _partition_count: int):
        """Benchmark the current re-scan-on-load behavior."""
        self.current_dataset.load()

    def time_load_cached(self, _filesystem: str, _partition_count: int):
        """Benchmark the old cached load behavior for comparison."""
        self.cached_dataset.load()

    def time_repeated_load_current(self, _filesystem: str, _partition_count: int):
        """Benchmark repeated calls to load() on the current implementation."""
        for _ in range(REPEAT_COUNT):
            self.current_dataset.load()

    def time_repeated_load_cached(self, _filesystem: str, _partition_count: int):
        """Benchmark repeated calls to load() on the cached implementation."""
        for _ in range(REPEAT_COUNT):
            self.cached_dataset.load()
