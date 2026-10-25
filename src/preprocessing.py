"""
Module A (Part 2) — Leakage-Free Preprocessing, Stratified Splitting, and Class Balancing.

Key guarantees:
- Duplicate rows are removed prior to train/test splitting.
- Train/test split occurs BEFORE fitting median imputers, categorical encoders, or samplers.
- Target labels and post-outcome/identifier fields are strictly excluded from input features.
- SMOTE-ENN (or SMOTE fallback for very small minority classes) is applied ONLY to training data.
- Encoded class ID <-> original label mapping is preserved and serialized.
- Inference transformation gracefully handles unknown columns and missing features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from config.settings import DEFAULT_RANDOM_SEED, LEAKAGE_AND_ID_COLUMNS


@dataclass
class SplitDataBundle:
    """Container for leak-free preprocessed train and test partitions."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    label_mapping: Dict[int, str]
    inverse_label_mapping: Dict[str, int]
    train_class_counts_before: Dict[str, int]
    train_class_counts_after: Dict[str, int]
    test_class_counts: Dict[str, int]
    resampling_method_applied: str
    stratified_split_used: bool
    duplicates_removed: int
    preprocessing_notes: List[str] = field(default_factory=list)


class DatasetPreprocessor:
    """
    Stateful preprocessor fitted solely on the training partition and reused at inference.
    """

    def __init__(self, target_col: str = "Attack_type") -> None:
        self.target_col = target_col
        self.feature_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numeric_imputer: Optional[SimpleImputer] = None
        self.categorical_encoder: Optional[OrdinalEncoder] = None
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.id_to_label: Dict[int, str] = {}
        self.label_to_id: Dict[str, int] = {}
        self.numeric_medians: Dict[str, float] = {}
        self.is_fitted: bool = False

    def _select_valid_feature_columns(self, df: pd.DataFrame) -> List[str]:
        leakage_lower = {c.lower() for c in LEAKAGE_AND_ID_COLUMNS}
        valid_cols = []
        for col in df.columns:
            if col == self.target_col:
                continue
            if col in LEAKAGE_AND_ID_COLUMNS or col.lower() in leakage_lower:
                continue
            valid_cols.append(col)
        if not valid_cols:
            raise ValueError("No valid feature columns remain after removing target and leakage/ID columns.")
        return valid_cols

    def fit(self, X_train_raw: pd.DataFrame, y_train_raw: pd.Series) -> "DatasetPreprocessor":
        """Fit imputers, categorical encoders, and label encoder on the training split ONLY."""
        self.feature_columns = self._select_valid_feature_columns(X_train_raw)
        X_sub = X_train_raw[self.feature_columns].copy()

        self.numeric_columns = []
        self.categorical_columns = []

        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(X_sub[col]):
                self.numeric_columns.append(col)
            else:
                # Check if string column is actually numeric values with stray strings
                coerced = pd.to_numeric(X_sub[col], errors="coerce")
                non_na_ratio = coerced.notna().mean() if len(coerced) > 0 else 0.0
                if non_na_ratio > 0.90:
                    X_sub[col] = coerced
                    self.numeric_columns.append(col)
                else:
                    self.categorical_columns.append(col)

        # Replace +/-inf with NaN before fitting median imputer
        if self.numeric_columns:
            num_df = X_sub[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
            num_df = num_df.replace([np.inf, -np.inf], np.nan)
            # If a column is 100% NaN in training, fill with 0.0 so SimpleImputer doesn't drop it
            for col in self.numeric_columns:
                if num_df[col].isna().all():
                    num_df[col] = 0.0
            self.numeric_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            self.numeric_imputer.fit(num_df)
            for idx, col in enumerate(self.numeric_columns):
                med_val = float(self.numeric_imputer.statistics_[idx])
                self.numeric_medians[col] = 0.0 if np.isnan(med_val) else med_val

        if self.categorical_columns:
            cat_df = X_sub[self.categorical_columns].fillna("MISSING").astype(str)
            self.categorical_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            )
            self.categorical_encoder.fit(cat_df)

        # Fit label encoder
        y_clean = y_train_raw.astype(str).to_numpy()
        self.label_encoder.fit(y_clean)
        self.id_to_label = {int(i): str(lbl) for i, lbl in enumerate(self.label_encoder.classes_)}
        self.label_to_id = {str(lbl): int(i) for i, lbl in enumerate(self.label_encoder.classes_)}

        self.is_fitted = True
        return self

    def transform_features(
        self,
        df_raw: pd.DataFrame,
        return_warnings: bool = False,
    ) -> Tuple[pd.DataFrame, List[str]] | pd.DataFrame:
        """
        Transform raw records into the exact numeric feature matrix expected by the model.
        Gracefully handles missing required features (fills with training medians) and
        ignores unknown extra columns.
        """
        if not self.is_fitted:
            raise RuntimeError("DatasetPreprocessor must be fitted before calling transform_features().")

        notes: List[str] = []
        if not isinstance(df_raw, pd.DataFrame):
            df_raw = pd.DataFrame(df_raw)

        missing_cols = [c for c in self.feature_columns if c not in df_raw.columns]
        extra_cols = [
            c for c in df_raw.columns
            if c not in self.feature_columns and c != self.target_col and c not in LEAKAGE_AND_ID_COLUMNS
        ]

        if missing_cols:
            notes.append(
                f"Missing {len(missing_cols)} required feature column(s) ({missing_cols[:5]}); filled with training baseline defaults."
            )
        if extra_cols:
            notes.append(
                f"Ignored {len(extra_cols)} unknown column(s) not present during training ({extra_cols[:5]})."
            )

        # Construct aligned DataFrame
        aligned = pd.DataFrame(index=df_raw.index)
        for col in self.feature_columns:
            if col in df_raw.columns:
                aligned[col] = df_raw[col]
            else:
                aligned[col] = self.numeric_medians.get(col, 0.0) if col in self.numeric_columns else "MISSING"

        out = pd.DataFrame(index=df_raw.index)

        if self.numeric_columns and self.numeric_imputer is not None:
            num_df = aligned[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
            num_df = num_df.replace([np.inf, -np.inf], np.nan)
            imputed_arr = self.numeric_imputer.transform(num_df)
            for idx, col in enumerate(self.numeric_columns):
                out[col] = imputed_arr[:, idx].astype(np.float64)

        if self.categorical_columns and self.categorical_encoder is not None:
            cat_df = aligned[self.categorical_columns].fillna("MISSING").astype(str)
            encoded_arr = self.categorical_encoder.transform(cat_df)
            for idx, col in enumerate(self.categorical_columns):
                out[col] = encoded_arr[:, idx].astype(np.float64)

        # Preserve exact column order
        out = out[self.feature_columns]

        if return_warnings:
            return out, notes
        return out

    def encode_labels(self, y_raw: pd.Series | np.ndarray | List[Any]) -> np.ndarray:
        """Encode string/raw labels into integer IDs; unseen labels map to -1."""
        y_str = pd.Series(y_raw).astype(str)
        return np.array([self.label_to_id.get(val, -1) for val in y_str], dtype=np.int64)

    def decode_labels(self, y_ids: np.ndarray | List[int]) -> List[str]:
        """Decode integer class IDs back to original string class names."""
        return [self.id_to_label.get(int(i), f"Unknown_Class_{i}") for i in y_ids]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "DatasetPreprocessor":
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {path}")
        obj = joblib.load(path)
        if not isinstance(obj, DatasetPreprocessor):
            raise TypeError(f"Artifact at {path} is not a valid DatasetPreprocessor.")
        return obj


def apply_training_resampling(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    method: str = "smote_enn",
    random_state: int = DEFAULT_RANDOM_SEED,
    max_samples_per_class: int = 350,
) -> Tuple[pd.DataFrame, np.ndarray, str]:
    """
    Apply class-imbalance resampling STRICTLY to the training split.
    Supports:
    - 'none': keep original training distribution.
    - 'smote_enn': SMOTE + Edited Nearest Neighbours (with safe k_neighbors adaptation
      when minority classes are small, or capped oversampling so runtime stays fast on laptops).
    - 'smote': SMOTE oversampling of minority classes up to a balanced cap.
    """
    if method is None or method.lower() == "none":
        return X_train, y_train, "None (Original training distribution + class_weight='balanced')"

    unique_classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min())
    max_count = int(counts.max())

    if len(unique_classes) < 2 or min_count < 3:
        return (
            X_train,
            y_train,
            f"Skipped resampling (minimum class has {min_count} samples, requires >= 3); using class_weight='balanced'",
        )

    try:
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours

        k_neighbors = min(5, max(1, min_count - 1))
        # Target a reasonable minority lift without exploding dataset size on a laptop
        target_min = min(max_count, max(int(np.median(counts)), min(max_samples_per_class, max_count)))
        sampling_strategy = {
            int(cls): max(int(cnt), target_min)
            for cls, cnt in zip(unique_classes, counts)
        }

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )

        if method.lower() == "smote_enn":
            enn = EditedNearestNeighbours(n_neighbors=min(3, k_neighbors), sampling_strategy="majority")
            sampler = SMOTEENN(smote=smote, enn=enn, random_state=random_state)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            # Ensure no class was completely wiped out by ENN
            if len(np.unique(y_res)) == len(unique_classes):
                return (
                    pd.DataFrame(X_res, columns=X_train.columns),
                    y_res,
                    f"SMOTE-ENN (training-only, k={k_neighbors})",
                )

        # Fallback to pure SMOTE if ENN removed an entire ultra-small class or if 'smote' requested
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return (
            pd.DataFrame(X_res, columns=X_train.columns),
            y_res,
            f"SMOTE (training-only, k={k_neighbors})",
        )
    except Exception as exc:
        warnings.warn(f"Resampling fell back to class weighting due to: {exc}")
        return X_train, y_train, f"Fallback to class_weight='balanced' ({exc})"


def prepare_train_test_data(
    df: pd.DataFrame,
    target_col: str = "Attack_type",
    test_size: float = 0.20,
    resampling_method: str = "smote_enn",
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Tuple[SplitDataBundle, DatasetPreprocessor]:
    """
    Complete leak-free data preparation pipeline:
    1. Drop rows with missing target and remove duplicate rows.
    2. Partition into train/test sets (stratified when all classes have >= 2 samples).
    3. Fit DatasetPreprocessor exclusively on training partition.
    4. Transform train and test partitions.
    5. Optionally apply SMOTE-ENN strictly on the training partition.
    """
    if target_col not in df.columns:
        raise ValueError(f"Selected target column '{target_col}' not found in dataset columns.")

    notes: List[str] = []
    clean_df = df.dropna(subset=[target_col]).copy()
    before_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before_dedup - len(clean_df)
    if duplicates_removed > 0:
        notes.append(f"Removed {duplicates_removed} duplicate rows prior to train/test split.")

    y_raw = clean_df[target_col].astype(str)
    vc = y_raw.value_counts()
    if len(vc) < 2:
        raise ValueError(f"Target column '{target_col}' must contain at least 2 distinct classes.")

    # Check if stratified split is possible (every class needs >= 2 samples, and test_size * N >= n_classes)
    min_class_samples = int(vc.min())
    can_stratify = min_class_samples >= 2 and int(len(clean_df) * test_size) >= len(vc)

    if can_stratify:
        train_df, test_df = train_test_split(
            clean_df,
            test_size=test_size,
            stratify=y_raw,
            random_state=random_state,
        )
        notes.append(f"Performed stratified train/test split ({int((1 - test_size)*100)}% / {int(test_size*100)}%).")
    else:
        train_df, test_df = train_test_split(
            clean_df,
            test_size=test_size,
            stratify=None,
            random_state=random_state,
        )
        notes.append(
            f"Fell back to non-stratified split because minority class has only {min_class_samples} sample(s)."
        )

    preprocessor = DatasetPreprocessor(target_col=target_col)
    preprocessor.fit(train_df, train_df[target_col])

    X_train_clean = preprocessor.transform_features(train_df)
    X_test_clean = preprocessor.transform_features(test_df)

    y_train_enc = preprocessor.encode_labels(train_df[target_col])
    y_test_enc = preprocessor.encode_labels(test_df[target_col])

    # Handle rare case where non-stratified split placed a 1-sample class only in test_df
    valid_test_mask = y_test_enc >= 0
    if not np.all(valid_test_mask):
        dropped_unseen = int((~valid_test_mask).sum())
        X_test_clean = X_test_clean.loc[valid_test_mask].reset_index(drop=True)
        y_test_enc = y_test_enc[valid_test_mask]
        notes.append(f"Excluded {dropped_unseen} test record(s) belonging to a class with 0 training examples.")

    # Record class distribution before resampling
    train_counts_before = {
        preprocessor.id_to_label[int(c)]: int(cnt)
        for c, cnt in zip(*np.unique(y_train_enc, return_counts=True))
    }

    # Resample training partition ONLY
    X_train_res, y_train_res, applied_resampling = apply_training_resampling(
        X_train_clean.reset_index(drop=True),
        y_train_enc,
        method=resampling_method,
        random_state=random_state,
    )
    notes.append(f"Training imbalance handling: {applied_resampling}.")

    train_counts_after = {
        preprocessor.id_to_label[int(c)]: int(cnt)
        for c, cnt in zip(*np.unique(y_train_res, return_counts=True))
    }
    test_counts = {
        preprocessor.id_to_label[int(c)]: int(cnt)
        for c, cnt in zip(*np.unique(y_test_enc, return_counts=True))
    }

    bundle = SplitDataBundle(
        X_train=X_train_res,
        X_test=X_test_clean.reset_index(drop=True),
        y_train=y_train_res,
        y_test=y_test_enc,
        feature_names=preprocessor.feature_columns,
        label_mapping=preprocessor.id_to_label,
        inverse_label_mapping=preprocessor.label_to_id,
        train_class_counts_before=train_counts_before,
        train_class_counts_after=train_counts_after,
        test_class_counts=test_counts,
        resampling_method_applied=applied_resampling,
        stratified_split_used=can_stratify,
        duplicates_removed=duplicates_removed,
        preprocessing_notes=notes,
    )
    return bundle, preprocessor
