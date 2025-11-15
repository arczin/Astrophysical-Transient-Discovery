#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
import json


DEFAULT_DATA_DIR = Path(__file__).resolve().parent


class PipelineDataValidator:
    """Validates that the uploaded data is consistent, complete, and ready for a time-series pipeline."""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)

    def validate_all(self) -> Tuple[bool, Dict]:
        print("Running dataset validation...")

        detections_df = pd.read_csv(self.data_dir / "detections.csv")
        injections_df = pd.read_csv(self.data_dir / "injections.csv")
        meta_df = pd.read_csv(self.data_dir / "object_meta.csv")

        results = {}
        all_valid = True

        # Required structure
        required_cols = ['object_id', 'epoch_day', 'mag', 'flux', 'mag_err', 'flux_err']
        missing = [col for col in required_cols if col not in detections_df.columns]

        if missing:
            all_valid = False
        results['columns_ok'] = len(missing) == 0

        # NaN check
        critical = detections_df[['object_id', 'epoch_day', 'mag', 'flux']]
        nan_count = critical.isna().sum().sum()
        results['no_nans'] = nan_count == 0

        # Attempt time-series pivot
        try:
            ts_matrix = detections_df.pivot_table(
                index='object_id',
                columns='epoch_day',
                values='mag',
                aggfunc='mean'
            )

            sparsity = ts_matrix.isna().sum().sum() / ts_matrix.size

            results['matrix_ok'] = True
            results['matrix_shape'] = ts_matrix.shape
            results['sparsity'] = sparsity
        except Exception:
            results['matrix_ok'] = False
            all_valid = False

        # Label consistency
        inj_from_dets = (
            detections_df[detections_df['is_injection'] == 1]['injection_id']
            .dropna()
            .unique()
        )
        inj_from_table = injections_df['injection_id'].unique()

        unmatched = set(inj_from_dets) - set(inj_from_table)
        results['labels_match'] = len(unmatched) == 0

        # Sanity checks
        mag_outliers = ((detections_df['mag'] < 10) | (detections_df['mag'] > 30)).sum()
        neg_flux = (detections_df['flux'] < 0).sum()

        results['reasonable_ranges'] = (mag_outliers == 0 and neg_flux == 0)

        print("Validation complete.")
        return all_valid, results


def create_pipeline_ready_data(data_dir: Path = DEFAULT_DATA_DIR, output_dir: Optional[Path] = None):
    print("Creating pipeline-ready data files...")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir is not None else data_dir / "outputs"

    output_dir.mkdir(parents=True, exist_ok=True)

    detections_df = pd.read_csv(data_dir / "detections.csv")

    # Build time series matrix
    ts = detections_df.pivot_table(
        index='object_id',
        columns='epoch_day',
        values='mag',
        aggfunc='mean'
    )
    ts = ts.ffill(axis=1).bfill(axis=1)

    ts_path = output_dir / "time_series_matrix.csv"
    ts.to_csv(ts_path)

    # Build label matrix
    labels = detections_df.pivot_table(
        index='object_id',
        columns='epoch_day',
        values='is_injection',
        aggfunc='max'
    ).fillna(0)

    labels_path = output_dir / "anomaly_labels.csv"
    labels.to_csv(labels_path)

    # Metadata summary
    metadata = {
        'n_objects': ts.shape[0],
        'n_timestamps': ts.shape[1],
        'anomaly_rate': labels.values.mean(),
        'data_type': 'time_series',
        'source': 'uploaded_dataset'
    }

    meta_path = output_dir / "data_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Pipeline-ready data created.")

    return ts, labels, metadata


if __name__ == "__main__":
    validator = PipelineDataValidator()
    is_valid, results = validator.validate_all()

    print("Validation statistics:")
    for key, value in results.items():
        print(f"  {key}: {value}")

    print(f"Ready for modeling: {is_valid}")
    print("Done.")
