from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from suspicious_reviews.preprocessing import preprocess_reviews
from suspicious_reviews.io import save_csv


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "booking_reviews_traindataset_output.csv"
)

EMBEDDING_PATH = (
    ROOT
    / "models"
    / "review_embeddings_minilm.npy"
)

OUTPUT = (
    ROOT
    / "results"
    / "phase1_foldwise_projected"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 42

N_SPLITS = 5

SIMILARITY_THRESHOLD = 0.80

LABEL_SIMILARITY_COUNT = 2

LABEL_MAX_SIMILARITY = 0.85

PAGERANK_QUANTILE = 0.95


# ============================================================
# FEATURES
# ============================================================

BASE_FEATURES = [
    "review_score_numeric",
    "token_count",
    "character_count",
]


GRAPH_FEATURES = [
    "degree_centrality",
    "pagerank",
    "betweenness_centrality",
    "clustering_coefficient",
    "triangle_count",
    "eigenvector_centrality",
    "similarity_count",
    "max_similarity",
    "community_size",
    "component_size",
]


# Strict leakage-controlled graph features.
#
# These features are NOT directly used in the primary
# predictive configuration when they are related to the
# graph properties used in the weak-label rules.
GRAPH_FEATURES_NO_LEAKAGE = [
    "betweenness_centrality",
    "clustering_coefficient",
    "triangle_count",
    "eigenvector_centrality",
    "component_size",
]


MODEL_FEATURES = (
    BASE_FEATURES
    + GRAPH_FEATURES_NO_LEAKAGE
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> tuple[pd.DataFrame, np.ndarray]:

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    if not EMBEDDING_PATH.exists():

        raise FileNotFoundError(
            f"Embeddings not found:\n{EMBEDDING_PATH}"
        )

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(
        f"Rows before preprocessing: {len(df)}"
    )

    # Use the project's existing preprocessing function.
    df = preprocess_reviews(df)

    print(
        f"Rows after preprocessing: {len(df)}"
    )

    print("\nLoading MiniLM embeddings...")

    embeddings = np.load(
        EMBEDDING_PATH
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    if len(df) != len(embeddings):

        raise ValueError(
            "Dataset rows and embedding rows do not match.\n"
            f"Dataset rows: {len(df)}\n"
            f"Embeddings: {len(embeddings)}"
        )

    return df, embeddings


# ============================================================
# COSINE-SIMILARITY EDGES
# ============================================================

def build_training_edges(
    review_ids: np.ndarray,
    embeddings: np.ndarray,
) -> list[tuple[int, int, float]]:

    """
    Build edges ONLY among training reviews.

    No validation reviews are involved here.
    """

    similarity_matrix = cosine_similarity(
        embeddings,
        embeddings,
    )

    edges = []

    n = len(review_ids)

    for i in range(n):

        for j in range(
            i + 1,
            n,
        ):

            similarity = float(
                similarity_matrix[i, j]
            )

            if (
                similarity
                >= SIMILARITY_THRESHOLD
            ):

                edges.append(
                    (
                        int(review_ids[i]),
                        int(review_ids[j]),
                        similarity,
                    )
                )

    return edges


def build_validation_training_edges(
    train_ids: np.ndarray,
    train_embeddings: np.ndarray,
    validation_ids: np.ndarray,
    validation_embeddings: np.ndarray,
) -> list[tuple[int, int, float]]:

    """
    Build ONLY validation -> training edges.

    Validation -> validation similarities are never computed.
    """

    similarity_matrix = cosine_similarity(
        validation_embeddings,
        train_embeddings,
    )

    edges = []

    for i in range(
        len(validation_ids)
    ):

        validation_id = int(
            validation_ids[i]
        )

        for j in range(
            len(train_ids)
        ):

            similarity = float(
                similarity_matrix[i, j]
            )

            if (
                similarity
                >= SIMILARITY_THRESHOLD
            ):

                train_id = int(
                    train_ids[j]
                )

                edges.append(
                    (
                        validation_id,
                        train_id,
                        similarity,
                    )
                )

    return edges


# ============================================================
# CREATE GRAPH
# ============================================================

def create_graph(
    reviews: pd.DataFrame,
    edges: list[tuple[int, int, float]],
) -> nx.Graph:

    graph = nx.Graph()

    for row in reviews.itertuples(
        index=False
    ):

        graph.add_node(
            int(row.review_id)
        )

    for (
        source,
        target,
        similarity,
    ) in edges:

        graph.add_edge(
            source,
            target,
            weight=similarity,
            similarity_score=similarity,
        )

    return graph


# ============================================================
# COMMUNITY DETECTION
# ============================================================

def detect_communities(
    graph: nx.Graph,
) -> list[set[int]]:

    if graph.number_of_edges() == 0:

        return [
            {node}
            for node in graph.nodes
        ]

    try:

        communities = list(
            nx.algorithms.community
            .louvain_communities(
                graph,
                weight="weight",
                seed=SEED,
            )
        )

        return communities

    except Exception:

        return list(
            nx.connected_components(
                graph
            )
        )


# ============================================================
# GRAPH FEATURE EXTRACTION
# ============================================================

def compute_graph_features(
    graph: nx.Graph,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # Degree Centrality
    # --------------------------------------------------------

    degree_centrality = (
        nx.degree_centrality(graph)
        if graph.number_of_nodes() > 0
        else {}
    )

    # --------------------------------------------------------
    # PageRank
    # --------------------------------------------------------

    if graph.number_of_edges() > 0:

        pagerank = nx.pagerank(
            graph,
            weight="weight",
        )

    else:

        pagerank = {
            node: 0.0
            for node in graph.nodes
        }

    # --------------------------------------------------------
    # Betweenness
    # --------------------------------------------------------

    if graph.number_of_edges() > 0:

        betweenness = (
            nx.betweenness_centrality(
                graph,
                weight="weight",
            )
        )

    else:

        betweenness = {
            node: 0.0
            for node in graph.nodes
        }

    # --------------------------------------------------------
    # Clustering
    # --------------------------------------------------------

    if graph.number_of_edges() > 0:

        clustering = nx.clustering(
            graph,
            weight="weight",
        )

    else:

        clustering = {
            node: 0.0
            for node in graph.nodes
        }

    # --------------------------------------------------------
    # Triangle count
    # --------------------------------------------------------

    if graph.number_of_edges() > 0:

        triangles = nx.triangles(
            graph
        )

    else:

        triangles = {
            node: 0
            for node in graph.nodes
        }

    # --------------------------------------------------------
    # Eigenvector centrality
    # --------------------------------------------------------

    if graph.number_of_edges() > 0:

        try:

            eigenvector = (
                nx.eigenvector_centrality(
                    graph,
                    max_iter=1000,
                    weight="weight",
                )
            )

        except Exception:

            eigenvector = {
                node: 0.0
                for node in graph.nodes
            }

    else:

        eigenvector = {
            node: 0.0
            for node in graph.nodes
        }

    # --------------------------------------------------------
    # Connected component size
    # --------------------------------------------------------

    components = list(
        nx.connected_components(graph)
    )

    component_size = {}

    for component in components:

        size = len(component)

        for node in component:

            component_size[node] = size

    # --------------------------------------------------------
    # Louvain communities
    # --------------------------------------------------------

    communities = detect_communities(
        graph
    )

    community_id = {}

    community_size = {}

    for cid, community in enumerate(
        communities
    ):

        for node in community:

            community_id[node] = cid

            community_size[node] = (
                len(community)
            )

    # --------------------------------------------------------
    # Similarity count
    # --------------------------------------------------------

    similarity_count = {
        node: int(
            graph.degree(node)
        )
        for node in graph.nodes
    }

    # --------------------------------------------------------
    # Maximum similarity
    # --------------------------------------------------------

    max_similarity = {}

    for node in graph.nodes:

        values = [
            float(
                data.get(
                    "weight",
                    0.0,
                )
            )
            for _, _, data
            in graph.edges(
                node,
                data=True,
            )
        ]

        max_similarity[node] = (
            max(values)
            if values
            else 0.0
        )

    # --------------------------------------------------------
    # Construct feature table
    # --------------------------------------------------------

    rows = []

    for node in graph.nodes:

        rows.append(
            {
                "review_id": int(node),

                "degree_centrality":
                    degree_centrality.get(
                        node,
                        0.0,
                    ),

                "pagerank":
                    pagerank.get(
                        node,
                        0.0,
                    ),

                "betweenness_centrality":
                    betweenness.get(
                        node,
                        0.0,
                    ),

                "clustering_coefficient":
                    clustering.get(
                        node,
                        0.0,
                    ),

                "triangle_count":
                    triangles.get(
                        node,
                        0,
                    ),

                "eigenvector_centrality":
                    eigenvector.get(
                        node,
                        0.0,
                    ),

                "similarity_count":
                    similarity_count.get(
                        node,
                        0,
                    ),

                "max_similarity":
                    max_similarity.get(
                        node,
                        0.0,
                    ),

                "community_id":
                    community_id.get(
                        node,
                        -1,
                    ),

                "community_size":
                    community_size.get(
                        node,
                        1,
                    ),

                "component_size":
                    component_size.get(
                        node,
                        1,
                    ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("review_id")
        .reset_index(drop=True)
    )


# ============================================================
# WEAK LABEL GENERATION
# ============================================================

def generate_labels(
    feature_df: pd.DataFrame,
    pagerank_cutoff: float,
) -> pd.Series:

    condition_one = (
        (
            feature_df[
                "similarity_count"
            ]
            >= LABEL_SIMILARITY_COUNT
        )
        &
        (
            feature_df[
                "max_similarity"
            ]
            >= LABEL_MAX_SIMILARITY
        )
    )

    condition_two = (
        (
            feature_df[
                "community_size"
            ]
            >= 3
        )
        &
        (
            feature_df[
                "pagerank"
            ]
            >= pagerank_cutoff
        )
    )

    labels = (
        condition_one
        |
        condition_two
    ).astype(int)

    return labels


# ============================================================
# BUILD TRAINING GRAPH DATA
# ============================================================

def build_training_fold(
    train_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    fold_dir: Path,
):
    print(
        "\nBuilding TRAINING graph..."
    )

    train_ids = (
        train_df[
            "review_id"
        ]
        .to_numpy(
            dtype=int
        )
    )

    # --------------------------------------------------------
    # TRAINING-ONLY GRAPH
    # --------------------------------------------------------

    train_edges = (
        build_training_edges(
            train_ids,
            train_embeddings,
        )
    )

    train_graph = create_graph(
        train_df,
        train_edges,
    )

    print(
        "Training graph nodes:",
        train_graph.number_of_nodes(),
    )

    print(
        "Training graph edges:",
        train_graph.number_of_edges(),
    )

    # --------------------------------------------------------
    # Training graph features
    # --------------------------------------------------------

    train_features = (
        compute_graph_features(
            train_graph
        )
    )

    # --------------------------------------------------------
    # PageRank threshold comes ONLY from
    # the training graph.
    # --------------------------------------------------------

    pagerank_cutoff = float(
        train_features[
            "pagerank"
        ]
        .quantile(
            PAGERANK_QUANTILE
        )
    )

    # --------------------------------------------------------
    # Training weak labels
    # --------------------------------------------------------

    train_features[
        "suspicious_label"
    ] = generate_labels(
        train_features,
        pagerank_cutoff,
    )

    # --------------------------------------------------------
    # Merge features with training reviews
    # --------------------------------------------------------

    train_data = train_df.merge(
        train_features,
        on="review_id",
        how="left",
    )

    # --------------------------------------------------------
    # Save training artifacts
    # --------------------------------------------------------

    train_edges_df = pd.DataFrame(
        train_edges,
        columns=[
            "review_id_1",
            "review_id_2",
            "similarity_score",
        ],
    )

    save_csv(
        train_edges_df,
        fold_dir
        / "train_similar_reviews.csv",
    )

    save_csv(
        train_features,
        fold_dir
        / "train_graph_features.csv",
    )

    # Save the training weak labels separately.
    save_csv(
        train_features[
            [
                "review_id",
                "suspicious_label",
            ]
        ],
        fold_dir
        / "train_labels.csv",
    )

    return (
        train_data,
        train_graph,
        pagerank_cutoff,
        len(train_edges),
    )


# ============================================================
# STRICT VALIDATION PROJECTION
# ============================================================

def project_validation_reviews(
    train_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    val_df: pd.DataFrame,
    val_embeddings: np.ndarray,
    train_graph: nx.Graph,
    pagerank_cutoff: float,
    fold_dir: Path,
):
    """
    Strict validation projection.

    Every validation review is projected separately:

        fixed training graph
              +
        ONE validation review
              +
        that validation review's
        validation-to-training edges

    No other validation review is included.

    Therefore, validation reviews cannot affect one another.
    """

    train_ids = (
        train_df[
            "review_id"
        ]
        .to_numpy(
            dtype=int
        )
    )

    validation_ids = (
        val_df[
            "review_id"
        ]
        .to_numpy(
            dtype=int
        )
    )

    # --------------------------------------------------------
    # Compute ONLY validation -> training similarities.
    # --------------------------------------------------------

    cross_edges = (
        build_validation_training_edges(
            train_ids,
            train_embeddings,
            validation_ids,
            val_embeddings,
        )
    )

    # --------------------------------------------------------
    # Group cross edges by validation review.
    # --------------------------------------------------------

    edge_map = {
        int(review_id): []
        for review_id in validation_ids
    }

    for (
        validation_id,
        train_id,
        similarity,
    ) in cross_edges:

        edge_map[
            int(validation_id)
        ].append(
            (
                int(train_id),
                float(similarity),
            )
        )

    validation_feature_rows = []

    validation_label_rows = []

    # --------------------------------------------------------
    # Project EACH validation review independently.
    # --------------------------------------------------------

    total = len(validation_ids)

    print(
        "\nProjecting validation reviews individually..."
    )

    for position, validation_id in enumerate(
        validation_ids,
        start=1,
    ):

        # ----------------------------------------------------
        # Copy the fixed training graph.
        # ----------------------------------------------------

        projected_graph = (
            train_graph.copy()
        )

        # ----------------------------------------------------
        # Add ONLY THIS validation node.
        # ----------------------------------------------------

        projected_graph.add_node(
            int(validation_id)
        )

        # ----------------------------------------------------
        # Add only this validation review's
        # edges to training reviews.
        # ----------------------------------------------------

        for (
            train_id,
            similarity,
        ) in edge_map.get(
            int(validation_id),
            [],
        ):

            projected_graph.add_edge(
                int(validation_id),
                int(train_id),
                weight=similarity,
                similarity_score=similarity,
            )

        # ----------------------------------------------------
        # Safety check:
        # no other validation node may be present.
        # ----------------------------------------------------

        validation_id_set = set(
            int(x)
            for x in validation_ids
        )

        present_validation_nodes = (
            validation_id_set
            &
            set(
                int(x)
                for x in projected_graph.nodes
            )
        )

        present_validation_nodes.discard(
            int(validation_id)
        )

        if present_validation_nodes:

            raise RuntimeError(
                "Validation leakage detected.\n"
                "Other validation nodes found:\n"
                f"{sorted(present_validation_nodes)}"
            )

        # ----------------------------------------------------
        # Compute features for this projected graph.
        # ----------------------------------------------------

        projected_features = (
            compute_graph_features(
                projected_graph
            )
        )

        # ----------------------------------------------------
        # Keep ONLY this validation node.
        # ----------------------------------------------------

        validation_row = (
            projected_features[
                projected_features[
                    "review_id"
                ]
                == int(validation_id)
            ]
            .copy()
        )

        if validation_row.empty:

            raise RuntimeError(
                f"Could not compute features for "
                f"validation review {validation_id}."
            )

        validation_row = (
            validation_row
            .iloc[0]
            .to_dict()
        )

        validation_feature_rows.append(
            validation_row
        )

        # ----------------------------------------------------
        # Generate label for this validation review.
        #
        # The PageRank cutoff comes exclusively
        # from the training graph.
        # ----------------------------------------------------

        single_row_df = pd.DataFrame(
            [validation_row]
        )

        validation_label = generate_labels(
            single_row_df,
            pagerank_cutoff,
        )

        validation_label_rows.append(
            {
                "review_id":
                    int(validation_id),

                "suspicious_label":
                    int(
                        validation_label.iloc[0]
                    ),
            }
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            position == 1
            or
            position % 25 == 0
            or
            position == total
        ):

            print(
                f"  Projected "
                f"{position}/{total} validation reviews"
            )

    # --------------------------------------------------------
    # Validation feature dataframe
    # --------------------------------------------------------

    validation_features = (
        pd.DataFrame(
            validation_feature_rows
        )
        .sort_values(
            "review_id"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validation label dataframe
    # --------------------------------------------------------

    validation_labels = (
        pd.DataFrame(
            validation_label_rows
        )
    )

    # --------------------------------------------------------
    # Merge with validation reviews
    # --------------------------------------------------------

    validation_data = (
        val_df
        .merge(
            validation_features,
            on="review_id",
            how="left",
        )
        .merge(
            validation_labels,
            on="review_id",
            how="left",
        )
    )

    # --------------------------------------------------------
    # Fill missing numeric values
    # --------------------------------------------------------

    for column in GRAPH_FEATURES:

        if column in validation_data.columns:

            validation_data[
                column
            ] = (
                validation_data[
                    column
                ]
                .fillna(0)
            )

    validation_data[
        "suspicious_label"
    ] = (
        validation_data[
            "suspicious_label"
        ]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Save validation -> training edges
    # --------------------------------------------------------

    cross_edges_df = pd.DataFrame(
        cross_edges,
        columns=[
            "validation_review_id",
            "train_review_id",
            "similarity_score",
        ],
    )

    save_csv(
        cross_edges_df,
        fold_dir
        / "validation_to_train_edges.csv",
    )

    # --------------------------------------------------------
    # Explicit empty validation-validation file
    # --------------------------------------------------------

    validation_validation_df = pd.DataFrame(
        columns=[
            "validation_review_id_1",
            "validation_review_id_2",
            "similarity_score",
        ]
    )

    save_csv(
        validation_validation_df,
        fold_dir
        / "validation_to_validation_edges.csv",
    )

    # --------------------------------------------------------
    # Save validation projected features
    # --------------------------------------------------------

    save_csv(
        validation_features,
        fold_dir
        / "validation_projected_graph_features.csv",
    )

    # --------------------------------------------------------
    # Save validation labels
    # --------------------------------------------------------

    save_csv(
        validation_labels,
        fold_dir
        / "validation_labels.csv",
    )

    return (
        validation_data,
        len(cross_edges),
    )


# ============================================================
# XGBOOST MODEL
# ============================================================

def create_xgboost_model() -> XGBClassifier:

    return XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict:

    result = {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "roc_auc":
            np.nan,

        "pr_auc":
            np.nan,
    }

    if y_true.nunique() > 1:

        result["roc_auc"] = (
            roc_auc_score(
                y_true,
                y_probability,
            )
        )

        result["pr_auc"] = (
            average_precision_score(
                y_true,
                y_probability,
            )
        )

    return result


# ============================================================
# CROSS-VALIDATION MODEL EVALUATION
# ============================================================

def build_imbalance_experiments():
    """
    Return the eight imbalance-handling conditions used in the study.

    1. None
    2. Class Weight
    3. Random Oversampling
    4. SMOTE
    5. Borderline-SMOTE
    6. SVM-SMOTE
    7. KMeans-SMOTE
    8. ADASYN

    Every resampling operation is applied ONLY to X_train/y_train.
    Validation data are never resampled.
    """
    from sklearn.base import clone

    try:
        from imblearn.over_sampling import (
            RandomOverSampler,
            SMOTE,
            BorderlineSMOTE,
            SVMSMOTE,
            KMeansSMOTE,
            ADASYN,
        )
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for the eight imbalance "
            "experiments. Install it with: pip install imbalanced-learn"
        ) from exc

    return [
        {
            "name": "XGBoost",
            "short_name": "None",
            "resampler": None,
            "class_weight": False,
        },
        {
            "name": "XGBoost (Class Weight)",
            "short_name": "Class Weight",
            "resampler": None,
            "class_weight": True,
        },
        {
            "name": "XGBoost (Random Oversampling)",
            "short_name": "Random Oversampling",
            "resampler": RandomOverSampler(random_state=SEED),
            "class_weight": False,
        },
        {
            "name": "XGBoost (SMOTE)",
            "short_name": "SMOTE",
            "resampler": SMOTE(random_state=SEED),
            "class_weight": False,
        },
        {
            "name": "XGBoost (Borderline-SMOTE)",
            "short_name": "Borderline-SMOTE",
            "resampler": BorderlineSMOTE(
                random_state=SEED,
                kind="borderline-1",
            ),
            "class_weight": False,
        },
        {
            "name": "XGBoost (SVM-SMOTE)",
            "short_name": "SVM-SMOTE",
            "resampler": SVMSMOTE(random_state=SEED),
            "class_weight": False,
        },
        {
            "name": "XGBoost (KMeans-SMOTE)",
            "short_name": "KMeans-SMOTE",
            "resampler": KMeansSMOTE(
                random_state=SEED,
                cluster_balance_threshold=0.0,
            ),
            "class_weight": False,
        },
        {
            "name": "XGBoost (ADASYN)",
            "short_name": "ADASYN",
            "resampler": ADASYN(random_state=SEED),
            "class_weight": False,
        },
    ]


def calculate_detailed_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict:
    """Calculate imbalance-aware and minority-class metrics."""
    from sklearn.metrics import classification_report

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

        "class_0_precision": report["0"]["precision"],
        "class_0_recall": report["0"]["recall"],
        "class_0_f1": report["0"]["f1-score"],

        "class_1_precision": report["1"]["precision"],
        "class_1_recall": report["1"]["recall"],
        "class_1_f1": report["1"]["f1-score"],

        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],

        # Positive-class aliases retained for compatibility.
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),

        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }

    if y_true.nunique() > 1:
        result["roc_auc"] = roc_auc_score(
            y_true,
            y_probability,
        )
        result["pr_auc"] = average_precision_score(
            y_true,
            y_probability,
        )

    return result


def apply_imbalance_strategy(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    experiment: dict,
):
    """
    Apply the selected imbalance strategy to TRAINING data only.

    The returned X_fit/y_fit are the only data passed to model.fit().
    """
    X_fit = X_train
    y_fit = y_train

    if experiment["class_weight"]:
        negative_count = int((y_train == 0).sum())
        positive_count = int((y_train == 1).sum())

        if positive_count == 0:
            raise RuntimeError(
                "Cannot calculate class weight because the training "
                "fold contains no suspicious samples."
            )

        scale_pos_weight = negative_count / positive_count
        model.set_params(
            scale_pos_weight=scale_pos_weight
        )

    resampler = experiment["resampler"]

    if resampler is not None:
        # Fresh resampler for every fold.
        from sklearn.base import clone
        resampler = clone(resampler)
        X_fit, y_fit = resampler.fit_resample(
            X_train,
            y_train,
        )

    return X_fit, y_fit


def evaluate_one_imbalance_strategy(
    folds,
    experiment: dict,
):
    """
    Evaluate one imbalance strategy over the SAME strict fold-specific
    graph data.

    Graph construction is already completed separately inside each fold.
    This function only performs model training/evaluation.
    """
    fold_results = []
    oof_predictions = []

    model_name = experiment["name"]

    print("\n" + "=" * 70)
    print(f"IMBALANCE STRATEGY: {model_name}")
    print("=" * 70)

    for fold_number, (
        train_data,
        validation_data,
    ) in enumerate(folds, start=1):

        print("\n" + "-" * 60)
        print(f"{model_name} - Fold {fold_number}")
        print("-" * 60)

        X_train = (
            train_data[MODEL_FEATURES]
            .fillna(0)
        )

        y_train = (
            train_data["suspicious_label"]
            .astype(int)
        )

        X_validation = (
            validation_data[MODEL_FEATURES]
            .fillna(0)
        )

        y_validation = (
            validation_data["suspicious_label"]
            .astype(int)
        )

        if y_train.nunique() < 2:
            raise RuntimeError(
                f"Fold {fold_number}: training labels contain "
                "only one class."
            )

        if y_validation.nunique() < 2:
            raise RuntimeError(
                f"Fold {fold_number}: validation labels contain "
                "only one class."
            )

        model = create_xgboost_model()

        # --------------------------------------------------------
        # CRITICAL: imbalance handling is TRAINING-ONLY.
        # --------------------------------------------------------
        X_fit, y_fit = apply_imbalance_strategy(
            model,
            X_train,
            y_train,
            experiment,
        )

        model.fit(
            X_fit,
            y_fit,
        )

        # --------------------------------------------------------
        # Validation data remain completely untouched.
        # --------------------------------------------------------
        y_pred = model.predict(
            X_validation
        )

        y_probability = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        metrics = calculate_detailed_metrics(
            y_validation,
            y_pred,
            y_probability,
        )

        metrics.update(
            {
                "fold": fold_number,
                "model": model_name,
                "imbalance_method": experiment["short_name"],
                "train_reviews": len(train_data),
                "validation_reviews": len(validation_data),
                "train_suspicious": int(
                    y_train.sum()
                ),
                "validation_suspicious": int(
                    y_validation.sum()
                ),
                "resampled_train_reviews": len(X_fit),
                "resampled_train_suspicious": int(
                    np.asarray(y_fit).sum()
                ),
            }
        )

        fold_results.append(metrics)

        predictions = (
            validation_data[
                [
                    "review_id",
                    "hotel_name",
                    "review_text",
                    "suspicious_label",
                ]
            ]
            .copy()
        )

        predictions["fold"] = fold_number
        predictions["model"] = model_name
        predictions["imbalance_method"] = experiment["short_name"]
        predictions["predicted_label"] = y_pred
        predictions["prediction_probability"] = y_probability

        oof_predictions.append(
            predictions
        )

        print(
            f"Accuracy = {metrics['accuracy']:.4f}"
        )
        print(
            f"Balanced Accuracy = "
            f"{metrics['balanced_accuracy']:.4f}"
        )
        print(
            f"Class-1 Precision = "
            f"{metrics['class_1_precision']:.4f}"
        )
        print(
            f"Class-1 Recall = "
            f"{metrics['class_1_recall']:.4f}"
        )
        print(
            f"Class-1 F1 = "
            f"{metrics['class_1_f1']:.4f}"
        )
        print(
            f"ROC-AUC = {metrics['roc_auc']:.4f}"
        )
        print(
            f"PR-AUC = {metrics['pr_auc']:.4f}"
        )

    fold_df = (
        pd.DataFrame(fold_results)
        .sort_values("fold")
        .reset_index(drop=True)
    )

    oof_df = (
        pd.concat(
            oof_predictions,
            ignore_index=True,
        )
        .sort_values("review_id")
        .reset_index(drop=True)
    )

    pooled_y_true = (
        oof_df["suspicious_label"]
        .astype(int)
    )

    pooled_y_pred = (
        oof_df["predicted_label"]
        .astype(int)
        .to_numpy()
    )

    pooled_probability = (
        oof_df["prediction_probability"]
        .astype(float)
        .to_numpy()
    )

    pooled_metrics = calculate_detailed_metrics(
        pooled_y_true,
        pooled_y_pred,
        pooled_probability,
    )

    summary = {
        "model": model_name,
        "imbalance_method": experiment["short_name"],

        "pooled_accuracy":
            pooled_metrics["accuracy"],
        "pooled_balanced_accuracy":
            pooled_metrics["balanced_accuracy"],

        "pooled_class_1_precision":
            pooled_metrics["class_1_precision"],
        "pooled_class_1_recall":
            pooled_metrics["class_1_recall"],
        "pooled_class_1_f1":
            pooled_metrics["class_1_f1"],

        "pooled_macro_f1":
            pooled_metrics["macro_f1"],

        "pooled_roc_auc":
            pooled_metrics["roc_auc"],
        "pooled_pr_auc":
            pooled_metrics["pr_auc"],
    }

    # Add mean/std across folds for the paper.
    summary_metrics = [
        "accuracy",
        "balanced_accuracy",
        "class_1_precision",
        "class_1_recall",
        "class_1_f1",
        "macro_f1",
        "roc_auc",
        "pr_auc",
    ]

    for metric in summary_metrics:
        summary[f"mean_{metric}"] = (
            fold_df[metric].mean()
        )
        summary[f"std_{metric}"] = (
            fold_df[metric].std(ddof=1)
        )

    return fold_df, oof_df, summary


def select_final_imbalance_strategy(
    summary_df: pd.DataFrame,
) -> pd.Series:
    """
    Select the final configuration using the predefined hierarchy:

    1. suspicious-class F1
    2. Macro F1
    3. Balanced Accuracy
    4. suspicious-class Recall
    5. PR-AUC

    This avoids selecting a method based on overall accuracy alone.
    """
    ranked = summary_df.sort_values(
        by=[
            "pooled_class_1_f1",
            "pooled_macro_f1",
            "pooled_balanced_accuracy",
            "pooled_class_1_recall",
            "pooled_pr_auc",
        ],
        ascending=False,
    ).reset_index(drop=True)

    return ranked.iloc[0]


def retrain_final_model(
    folds,
    selected_summary: pd.Series,
):
    """
    Retrain the selected configuration on all source reviews.

    This function is intentionally separate from cross-validation.
    Cross-validation is used for model selection first; only then is
    the selected imbalance strategy applied to the complete dataset.
    """
    selected_method = selected_summary["imbalance_method"]

    experiment = None
    for candidate in build_imbalance_experiments():
        if candidate["short_name"] == selected_method:
            experiment = candidate
            break

    if experiment is None:
        raise RuntimeError(
            f"Could not find selected imbalance method: "
            f"{selected_method}"
        )

    all_train = pd.concat(
        [fold[0] for fold in folds],
        ignore_index=True,
    ).drop_duplicates(
        subset=["review_id"]
    )

    X_all = (
        all_train[MODEL_FEATURES]
        .fillna(0)
    )

    y_all = (
        all_train["suspicious_label"]
        .astype(int)
    )

    model = create_xgboost_model()

    X_fit, y_fit = apply_imbalance_strategy(
        model,
        X_all,
        y_all,
        experiment,
    )

    model.fit(
        X_fit,
        y_fit,
    )

    return model, experiment, all_train


def evaluate_all_imbalance_methods(
    folds,
):
    """
    Run all eight imbalance strategies using the exact same
    leakage-controlled folds.
    """
    all_summaries = []
    all_fold_results = []
    all_oof_predictions = []

    results_dir = OUTPUT / "imbalance_comparison"
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiments = build_imbalance_experiments()

    print("\n" + "=" * 80)
    print("EIGHT-METHOD IMBALANCE ABLATION")
    print("=" * 80)

    for experiment in experiments:
        fold_df, oof_df, summary = (
            evaluate_one_imbalance_strategy(
                folds,
                experiment,
            )
        )

        all_summaries.append(summary)
        all_fold_results.append(fold_df)
        all_oof_predictions.append(oof_df)

        safe_name = (
            experiment["short_name"]
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        save_csv(
            fold_df,
            results_dir
            / f"{safe_name}_fold_metrics.csv",
        )

        save_csv(
            oof_df,
            results_dir
            / f"{safe_name}_oof_predictions.csv",
        )

    summary_df = (
        pd.DataFrame(all_summaries)
        .sort_values(
            by=[
                "pooled_class_1_f1",
                "pooled_macro_f1",
                "pooled_balanced_accuracy",
                "pooled_class_1_recall",
                "pooled_pr_auc",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    fold_results_df = pd.concat(
        all_fold_results,
        ignore_index=True,
    )

    oof_results_df = pd.concat(
        all_oof_predictions,
        ignore_index=True,
    )

    save_csv(
        summary_df,
        results_dir
        / "eight_method_summary.csv",
    )

    save_csv(
        fold_results_df,
        results_dir
        / "eight_method_fold_metrics.csv",
    )

    save_csv(
        oof_results_df,
        results_dir
        / "eight_method_oof_predictions.csv",
    )

    selected = select_final_imbalance_strategy(
        summary_df
    )

    print("\n" + "=" * 80)
    print("FINAL IMBALANCE COMPARISON")
    print("=" * 80)

    display_columns = [
        "imbalance_method",
        "pooled_accuracy",
        "pooled_balanced_accuracy",
        "pooled_class_1_precision",
        "pooled_class_1_recall",
        "pooled_class_1_f1",
        "pooled_macro_f1",
        "pooled_roc_auc",
        "pooled_pr_auc",
        "mean_class_1_f1",
        "std_class_1_f1",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(index=False)
    )

    print("\nSelected final imbalance strategy:")
    print(
        f"  {selected['imbalance_method']}"
    )
    print(
        f"  Suspicious-class F1: "
        f"{selected['pooled_class_1_f1']:.4f}"
    )
    print(
        f"  Macro F1: "
        f"{selected['pooled_macro_f1']:.4f}"
    )
    print(
        f"  Balanced Accuracy: "
        f"{selected['pooled_balanced_accuracy']:.4f}"
    )
    print(
        f"  Suspicious-class Recall: "
        f"{selected['pooled_class_1_recall']:.4f}"
    )

    return (
        summary_df,
        fold_results_df,
        oof_results_df,
        selected,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "STRICT FOLD-WISE GRAPH EXPERIMENT"
    )

    print(
        "========================================"
    )

    print(
        "\nProtocol:"
    )

    print(
        "1. Split into 5 stratified folds."
    )

    print(
        "2. Build graph only from training reviews."
    )

    print(
        "3. Project each validation review individually."
    )

    print(
        "4. No validation-validation edges."
    )

    print(
        "5. All eight imbalance methods applied only to training data."
    )

    print(
        "========================================\n"
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df, embeddings = load_data()

    # --------------------------------------------------------
    # Existing weak labels are used ONLY to determine
    # stratified fold membership.
    #
    # The labels used for model evaluation are regenerated
    # from each fold-specific graph.
    # --------------------------------------------------------

    existing_features_path = (
        ROOT
        / "results"
        / "reviews_with_graph_features.csv"
    )

    if not existing_features_path.exists():

        raise FileNotFoundError(
            "The existing graph-feature file was not found:\n"
            f"{existing_features_path}\n\n"
            "This file is used only to obtain a stable "
            "stratification target for the five folds."
        )

    existing = pd.read_csv(
        existing_features_path
    )

    if (
        "review_id"
        not in existing.columns
    ):

        raise KeyError(
            "review_id is missing from "
            "reviews_with_graph_features.csv"
        )

    if (
        "suspicious_label"
        not in existing.columns
    ):

        raise KeyError(
            "suspicious_label is missing from "
            "reviews_with_graph_features.csv"
        )

    stratification_data = (
        existing[
            [
                "review_id",
                "suspicious_label",
            ]
        ]
        .drop_duplicates(
            "review_id"
        )
        .sort_values(
            "review_id"
        )
    )

    # --------------------------------------------------------
    # Verify review IDs match
    # --------------------------------------------------------

    data_ids = set(
        df[
            "review_id"
        ].astype(int)
    )

    label_ids = set(
        stratification_data[
            "review_id"
        ].astype(int)
    )

    if data_ids != label_ids:

        raise ValueError(
            "review_id sets in the processed dataset and "
            "existing label file do not match."
        )

    stratification_data = (
        stratification_data[
            "suspicious_label"
        ]
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Stratified five-fold split
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    folds = []

    protocol_rows = []

    # --------------------------------------------------------
    # Process each fold
    # --------------------------------------------------------

    for fold_number, (
        train_idx,
        validation_idx,
    ) in enumerate(
        cv.split(
            df,
            stratification_data,
        ),
        start=1,
    ):

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"FOLD {fold_number}"
        )

        print(
            "=" * 60
        )

        fold_dir = (
            OUTPUT
            / f"fold_{fold_number}"
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_df = (
            df.iloc[train_idx]
            .copy()
        )

        validation_df = (
            df.iloc[validation_idx]
            .copy()
        )

        train_embeddings = (
            embeddings[train_idx]
        )

        validation_embeddings = (
            embeddings[validation_idx]
        )

        # ----------------------------------------------------
        # Training graph
        # ----------------------------------------------------

        (
            train_data,
            train_graph,
            pagerank_cutoff,
            train_edge_count,
        ) = build_training_fold(
            train_df,
            train_embeddings,
            fold_dir,
        )

        # ----------------------------------------------------
        # Validation projection
        # ----------------------------------------------------

        (
            validation_data,
            validation_training_edge_count,
        ) = project_validation_reviews(
            train_df,
            train_embeddings,
            validation_df,
            validation_embeddings,
            train_graph,
            pagerank_cutoff,
            fold_dir,
        )

        # ----------------------------------------------------
        # Verify no validation-validation edges
        # ----------------------------------------------------

        validation_validation_path = (
            fold_dir
            / "validation_to_validation_edges.csv"
        )

        validation_validation_edges = pd.read_csv(
            validation_validation_path
        )

        validation_validation_count = len(
            validation_validation_edges
        )

        if (
            validation_validation_count
            != 0
        ):

            raise RuntimeError(
                f"Fold {fold_number} contains "
                f"{validation_validation_count} "
                "validation-validation edges."
            )

        # ----------------------------------------------------
        # Save fold-level protocol information
        # ----------------------------------------------------

        protocol_row = {

            "fold":
                fold_number,

            "train_reviews":
                len(train_data),

            "validation_reviews":
                len(validation_data),

            "train_suspicious":
                int(
                    train_data[
                        "suspicious_label"
                    ].sum()
                ),

            "validation_suspicious":
                int(
                    validation_data[
                        "suspicious_label"
                    ].sum()
                ),

            "training_graph_nodes":
                train_graph.number_of_nodes(),

            "training_graph_edges":
                train_edge_count,

            "validation_to_training_edges":
                validation_training_edge_count,

            "validation_to_validation_edges":
                validation_validation_count,

            "pagerank_training_cutoff":
                pagerank_cutoff,

            "similarity_threshold":
                SIMILARITY_THRESHOLD,
        }

        protocol_rows.append(
            protocol_row
        )

        print(
            f"\nTraining reviews: "
            f"{protocol_row['train_reviews']}"
        )

        print(
            f"Validation reviews: "
            f"{protocol_row['validation_reviews']}"
        )

        print(
            f"Training suspicious: "
            f"{protocol_row['train_suspicious']}"
        )

        print(
            f"Validation suspicious: "
            f"{protocol_row['validation_suspicious']}"
        )

        print(
            f"Training graph edges: "
            f"{protocol_row['training_graph_edges']}"
        )

        print(
            f"Validation -> training edges: "
            f"{protocol_row['validation_to_training_edges']}"
        )

        print(
            "Validation -> validation edges: "
            f"{protocol_row['validation_to_validation_edges']}"
        )

        folds.append(
            (
                train_data,
                validation_data,
            )
        )

    # --------------------------------------------------------
    # Save protocol summary
    # --------------------------------------------------------

    protocol_df = (
        pd.DataFrame(
            protocol_rows
        )
        .sort_values("fold")
    )

    save_csv(
        protocol_df,
        OUTPUT
        / "fold_protocol_summary.csv",
    )

    # --------------------------------------------------------
    # Evaluate all eight imbalance-handling strategies
    # under the SAME strict fold-wise graph protocol.
    # --------------------------------------------------------

    (
        summary_df,
        fold_results_df,
        oof_results_df,
        selected_strategy,
    ) = evaluate_all_imbalance_methods(
        folds
    )

    # Save the main summary using a stable filename.
    save_csv(
        summary_df,
        OUTPUT
        / "model_summary.csv",
    )

    # --------------------------------------------------------
    # Retrain the selected configuration on all source reviews.
    # --------------------------------------------------------

    final_model, final_experiment, final_training_data = (
        retrain_final_model(
            folds,
            selected_strategy,
        )
    )

    final_model_dir = (
        OUTPUT / "final_model"
    )
    final_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    import joblib

    joblib.dump(
        final_model,
        final_model_dir
        / "best_model.joblib",
    )

    save_csv(
        pd.DataFrame(
            [
                {
                    "selected_model":
                        "XGBoost",
                    "selected_imbalance_method":
                        final_experiment[
                            "short_name"
                        ],
                    "training_reviews":
                        len(final_training_data),
                    "training_suspicious":
                        int(
                            final_training_data[
                                "suspicious_label"
                            ].sum()
                        ),
                    "pooled_suspicious_f1":
                        selected_strategy[
                            "pooled_class_1_f1"
                        ],
                    "pooled_pr_auc":
                        selected_strategy[
                            "pooled_pr_auc"
                        ],
                }
            ]
        ),
        final_model_dir
        / "final_model_summary.csv",
    )

    # --------------------------------------------------------
    # Final JSON metadata
    # --------------------------------------------------------

    experiment_metadata = {

        "experiment":
            "Strict fold-wise projected graph evaluation",

        "number_of_reviews":
            len(df),

        "embedding_model":
            "all-MiniLM-L6-v2",

        "embedding_dimension":
            384,

        "similarity_threshold":
            SIMILARITY_THRESHOLD,

        "cross_validation":
            "Stratified 5-fold",

        "training_graph_protocol":
            "Graph constructed using only training reviews",

        "validation_protocol":
            "Each validation review projected individually "
            "onto the fixed training graph",

        "validation_validation_edges":
            0,

        "imbalance_protocol":
            "All eight imbalance strategies applied only to training folds",

        "imbalance_methods":
            [
                "None",
                "Class Weight",
                "Random Oversampling",
                "SMOTE",
                "Borderline-SMOTE",
                "SVM-SMOTE",
                "KMeans-SMOTE",
                "ADASYN",
            ],

        "primary_features":
            MODEL_FEATURES,

        "graph_features_excluded_from_primary_model":
            [
                "degree_centrality",
                "pagerank",
                "similarity_count",
                "max_similarity",
                "community_size",
            ],

        "results":
            summary_df.to_dict(orient="records"),
    }

    (
        OUTPUT
        / "experiment_metadata.json"
    ).write_text(
        json.dumps(
            experiment_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Display final result
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL FOLD-SPECIFIC RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nExperiment completed successfully."
    )

    print(
        f"Results saved to:\n{OUTPUT}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()