"""Baseline: TF-IDF + Random Forest with optional SMOTE."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    USE_SMOTE,
    SMOTE_K_NEIGHBORS,
    RANDOM_STATE,
)


def build_baseline_pipeline(smote: bool = USE_SMOTE):
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    if smote:
        pipeline = ImbPipeline(
            [
                ("tfidf", tfidf),
                ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS, n_jobs=-1)),
                ("clf", rf),
            ]
        )
    else:
        pipeline = Pipeline([("tfidf", tfidf), ("clf", rf)])
    return pipeline


def fit_baseline(pipeline, X_train: list[str], y_train: np.ndarray):
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_baseline(pipeline, X) -> np.ndarray:
    return pipeline.predict(X)


def predict_proba_baseline(pipeline, X) -> np.ndarray:
    return pipeline.predict_proba(X)
