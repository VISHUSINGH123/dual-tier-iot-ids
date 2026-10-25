"""UNSW-NB15 Data Preprocessor and Pearson Correlation Feature Selector.

Ported and modularized from the reference notebook pipeline.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class UNSWPreprocessor:
    """Handles end-to-end preprocessing, one-hot encoding, and correlation-based feature selection."""

    def __init__(self, raw_csv_path: str | Path, features_metadata_path: str | Path | None = None) -> None:
        self.raw_csv_path = Path(raw_csv_path)
        self.features_metadata_path = Path(features_metadata_path) if features_metadata_path else None
        self.scaler = MinMaxScaler()
        self.bin_encoder = LabelEncoder()
        self.multi_encoder = LabelEncoder()
        self.selected_bin_features: list[str] = []
        self.selected_multi_features: list[str] = []

    def clean_and_normalize(self) -> pd.DataFrame:
        """Drop missing values, apply one-hot-encoding to categorical fields, and MinMax scale."""
        data = pd.read_csv(self.raw_csv_path)
        data = data.dropna()

        # Categorical columns in UNSW-NB15
        categorical_cols = ["proto", "service", "state"]
        existing_cat = [col for col in categorical_cols if col in data.columns]

        data_cat = pd.get_dummies(data[existing_cat])
        data_numeric = data.drop(columns=existing_cat, errors="ignore")

        # Normalize numeric columns except targets ('label', 'attack_cat', 'id')
        feature_cols = [c for c in data_numeric.columns if c not in {"id", "label", "attack_cat"}]
        data_numeric[feature_cols] = self.scaler.fit_transform(data_numeric[feature_cols])

        # Merge one-hot encoded categorical columns
        combined = pd.concat([data_numeric, data_cat], axis=1)
        return combined

    def build_binary_dataset(self, df: pd.DataFrame, corr_threshold: float = 0.3) -> tuple[pd.DataFrame, list[str]]:
        """Extract features with Pearson correlation > threshold against binary target label."""
        bin_df = df.copy()
        if "attack_cat" in bin_df.columns:
            bin_df = bin_df.drop(columns=["attack_cat"])
        if "id" in bin_df.columns:
            bin_df = bin_df.drop(columns=["id"])

        if bin_df["label"].dtype == object:
            bin_df["label"] = self.bin_encoder.fit_transform(bin_df["label"])

        corr = bin_df.corr()
        corr_target = abs(corr["label"])
        selected = list(corr_target[corr_target > corr_threshold].index)
        self.selected_bin_features = selected
        return bin_df[selected], selected

    def build_multi_dataset(self, df: pd.DataFrame, corr_threshold: float = 0.3) -> tuple[pd.DataFrame, list[str]]:
        """Extract features with Pearson correlation > threshold against multiclass attack_cat target."""
        multi_df = df.copy()
        if "label" in multi_df.columns:
            multi_df = multi_df.drop(columns=["label"])
        if "id" in multi_df.columns:
            multi_df = multi_df.drop(columns=["id"])

        if "attack_cat" in multi_df.columns:
            multi_df["target_class"] = self.multi_encoder.fit_transform(multi_df["attack_cat"].astype(str))
            # One-hot encode attack_cat for correlation calculation
            dummies = pd.get_dummies(multi_df["attack_cat"], prefix="attack_cat")
            multi_with_dummies = pd.concat([multi_df.drop(columns=["attack_cat"]), dummies], axis=1)

            corr = multi_with_dummies.corr()
            corr_target = abs(corr["target_class"])
            selected = list(corr_target[corr_target > corr_threshold].index)
            # Ensure target_class is preserved
            if "target_class" not in selected:
                selected.append("target_class")
            self.selected_multi_features = selected
            return multi_with_dummies[selected], selected

        return multi_df, list(multi_df.columns)
