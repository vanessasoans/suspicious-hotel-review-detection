
"""
Phase 4 (v2): External cross-domain qualitative case studies

This version fixes two issues before the final paper run:

1. The final XGBoost model is trained on ALL 2,011 source reviews after
   the strict cross-validation/model-selection experiment. A temporary
   holdout is NOT used for the final external-analysis model.

2. Case-study selection explicitly includes:
      - up to 2 predicted-suspicious cases
      - up to 2 predicted-normal cases
      - 1 case closest to the 0.50 decision threshold
   whenever those groups are available.

The external dataset:
    data/processed/reviews_cleaned.csv

Expected source files:
    data/processed/booking_reviews_traindataset_output.csv
    models/review_embeddings_minilm.npy

Existing external embeddings from the previous run are reused when:
    results/phase4_external_case_studies/external_minilm_embeddings.npy

No external->external graph edges are constructed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
from sentence_transformers import SentenceTransformer

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost is required. Install with: pip install xgboost"
    ) from exc


RANDOM_STATE = 42
SIMILARITY_THRESHOLD = 0.80
PAGERANK_QUANTILE = 0.95
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BASE_FEATURES = [
    "review_score_numeric",
    "token_count",
    "character_count",
]

GRAPH_FEATURES_NO_LEAKAGE = [
    "betweenness_centrality",
    "clustering_coefficient",
    "triangle_count",
    "eigenvector_centrality",
    "component_size",
]

FEATURES = BASE_FEATURES + GRAPH_FEATURES_NO_LEAKAGE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=r"data/processed/reviews_cleaned.csv",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=r"data/processed/booking_reviews_traindataset_output.csv",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=r"models/review_embeddings_minilm.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"results/phase4_external_case_studies_v2",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--reuse-external-embeddings",
        type=str,
        default=r"results/phase4_external_case_studies/external_minilm_embeddings.npy",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
    )
    return parser.parse_args()


def build_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def prepare_source(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "review_id",
        "review_text",
        "clean_review_text",
        "review_score_numeric",
        "token_count",
        "character_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Source dataset missing columns: {sorted(missing)}"
        )

    result = df.copy().reset_index(drop=True)

    # Do NOT drop rows. The source embedding matrix is aligned to all
    # 2,011 source rows.
    result["clean_review_text"] = (
        result["clean_review_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    result["review_score_numeric"] = pd.to_numeric(
        result["review_score_numeric"], errors="coerce"
    ).fillna(0)
    result["token_count"] = pd.to_numeric(
        result["token_count"], errors="coerce"
    ).fillna(0)
    result["character_count"] = pd.to_numeric(
        result["character_count"], errors="coerce"
    ).fillna(0)

    return result


def validate_external(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "review_id",
        "hotel_name",
        "review_title",
        "review_text",
        "clean_review_text",
        "review_score_numeric",
        "token_count",
        "character_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"External dataset missing columns: {sorted(missing)}"
        )

    result = df.copy()

    result["clean_review_text"] = (
        result["clean_review_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    result["review_text"] = (
        result["review_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    result["review_score_numeric"] = pd.to_numeric(
        result["review_score_numeric"], errors="coerce"
    ).fillna(0)
    result["token_count"] = pd.to_numeric(
        result["token_count"], errors="coerce"
    ).fillna(0)
    result["character_count"] = pd.to_numeric(
        result["character_count"], errors="coerce"
    ).fillna(0)

    # The uploaded retained external file is already cleaned. Removing
    # empty cleaned text is only a safety check.
    result = result[
        result["clean_review_text"].str.len() > 0
    ].reset_index(drop=True)

    return result


def build_source_graph(embeddings: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    n = len(embeddings)

    graph.add_nodes_from(range(n))

    chunk_size = 256

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = embeddings[start:end] @ embeddings.T

        for local_i in range(end - start):
            i = start + local_i

            scores = block[local_i, i + 1:]
            hits = np.flatnonzero(
                scores >= SIMILARITY_THRESHOLD
            )

            for offset in hits:
                j = i + 1 + int(offset)
                graph.add_edge(
                    i,
                    j,
                    weight=float(block[local_i, j]),
                )

    return graph


def source_features_and_labels(
    graph: nx.Graph,
) -> tuple[pd.DataFrame, np.ndarray]:
    degree = nx.degree_centrality(graph)

    if graph.number_of_edges():
        betweenness = nx.betweenness_centrality(
            graph,
            weight="weight",
        )
        clustering = nx.clustering(
            graph,
            weight="weight",
        )
        triangles = nx.triangles(graph)

        try:
            eigenvector = nx.eigenvector_centrality(
                graph,
                max_iter=1000,
                weight="weight",
            )
        except Exception:
            eigenvector = {
                node: 0.0 for node in graph.nodes
            }
    else:
        betweenness = {
            node: 0.0 for node in graph.nodes
        }
        clustering = {
            node: 0.0 for node in graph.nodes
        }
        triangles = {
            node: 0 for node in graph.nodes
        }
        eigenvector = {
            node: 0.0 for node in graph.nodes
        }

    pagerank = nx.pagerank(
        graph,
        weight="weight",
    )

    similarity_count = {
        node: int(graph.degree(node))
        for node in graph.nodes
    }

    max_similarity = {
        node: max(
            (
                float(data.get("weight", 0.0))
                for _, _, data
                in graph.edges(node, data=True)
            ),
            default=0.0,
        )
        for node in graph.nodes
    }

    communities = nx.algorithms.community.louvain_communities(
        graph,
        weight="weight",
        seed=RANDOM_STATE,
    )

    community_size = {}
    for community in communities:
        for node in community:
            community_size[node] = len(community)

    component_size = {
        node: len(component)
        for component in nx.connected_components(graph)
        for node in component
    }

    pagerank_cutoff = float(
        np.quantile(
            list(pagerank.values()),
            PAGERANK_QUANTILE,
        )
    )

    labels = np.array(
        [
            int(
                (
                    similarity_count[node] >= 2
                    and max_similarity[node] >= 0.85
                )
                or (
                    community_size[node] >= 3
                    and pagerank[node] >= pagerank_cutoff
                )
            )
            for node in graph.nodes
        ],
        dtype=int,
    )

    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "graph_index": node,
                "degree_centrality": degree[node],
                "pagerank": pagerank[node],
                "betweenness_centrality": betweenness[node],
                "clustering_coefficient": clustering[node],
                "triangle_count": triangles[node],
                "eigenvector_centrality": eigenvector[node],
                "similarity_count": similarity_count[node],
                "max_similarity": max_similarity[node],
                "community_size": community_size[node],
                "component_size": component_size[node],
            }
        )

    return pd.DataFrame(rows), labels


def projection_features(
    external_embedding: np.ndarray,
    source_graph: nx.Graph,
    source_embeddings: np.ndarray,
) -> dict[str, float]:
    similarity = source_embeddings @ external_embedding

    neighbors = np.flatnonzero(
        similarity >= SIMILARITY_THRESHOLD
    )

    if len(neighbors) == 0:
        return {
            "betweenness_centrality": 0.0,
            "clustering_coefficient": 0.0,
            "triangle_count": 0.0,
            "eigenvector_centrality": 0.0,
            "component_size": 1.0,
            "projection_degree": 0.0,
            "projection_max_similarity": 0.0,
        }

    affected_nodes = set()

    for node in neighbors:
        affected_nodes.update(
            nx.node_connected_component(
                source_graph,
                int(node),
            )
        )

    projected = source_graph.subgraph(
        affected_nodes
    ).copy()

    external_node = -1
    projected.add_node(external_node)

    for node in neighbors:
        projected.add_edge(
            external_node,
            int(node),
            weight=float(similarity[node]),
        )

    betweenness = nx.betweenness_centrality(
        projected,
        weight="weight",
        normalized=True,
    ).get(external_node, 0.0)

    clustering = nx.clustering(
        projected,
        nodes=[external_node],
        weight="weight",
    ).get(external_node, 0.0)

    triangles = nx.triangles(
        projected,
        nodes=[external_node],
    ).get(external_node, 0)

    try:
        eigenvector = nx.eigenvector_centrality(
            projected,
            max_iter=1000,
            weight="weight",
        ).get(external_node, 0.0)
    except Exception:
        eigenvector = 0.0

    component = nx.node_connected_component(
        projected,
        external_node,
    )

    return {
        "betweenness_centrality": float(betweenness),
        "clustering_coefficient": float(clustering),
        "triangle_count": float(triangles),
        "eigenvector_centrality": float(eigenvector),
        "component_size": float(len(component)),
        "projection_degree": float(len(neighbors)),
        "projection_max_similarity": float(
            np.max(similarity[neighbors])
        ),
    }


def choose_candidates(
    external_embeddings: np.ndarray,
    source_embeddings: np.ndarray,
    candidate_pool: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = external_embeddings @ source_embeddings.T

    degree = (
        scores >= SIMILARITY_THRESHOLD
    ).sum(axis=1)

    max_similarity = scores.max(axis=1)

    k = max(30, candidate_pool // 2)

    high_degree = np.argsort(degree)[-k:]
    high_similarity = np.argsort(max_similarity)[-k:]
    low_degree = np.argsort(degree)[:k]
    random_state = np.random.default_rng(RANDOM_STATE)
    random_idx = random_state.choice(
        np.arange(len(external_embeddings)),
        size=min(k, len(external_embeddings)),
        replace=False,
    )

    pool = np.unique(
        np.concatenate(
            [
                high_degree,
                high_similarity,
                low_degree,
                random_idx,
            ]
        )
    )

    if len(pool) > candidate_pool:
        priority = (
            degree[pool] * 10.0
            + max_similarity[pool]
        )
        order = np.argsort(priority)[::-1]
        pool = pool[order[:candidate_pool]]

    return (
        pool.astype(int),
        scores,
        degree.astype(int),
        max_similarity.astype(float),
    )


def shap_waterfall(
    model: XGBClassifier,
    X_row: pd.DataFrame,
    output_path: Path,
    title: str,
) -> str:
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_row)

    values = np.asarray(explanation.values)
    if values.ndim == 2:
        values = values[0]

    base_value = float(
        np.asarray(
            explanation.base_values
        ).reshape(-1)[0]
    )

    vector = shap.Explanation(
        values=values,
        base_values=base_value,
        data=X_row.iloc[0].to_numpy(dtype=float),
        feature_names=X_row.columns.tolist(),
    )

    plt.figure(figsize=(9, 6))

    shap.plots.waterfall(
        vector,
        max_display=8,
        show=False,
    )

    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    ranked = sorted(
        zip(
            X_row.columns.tolist(),
            values.tolist(),
        ),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )

    return "; ".join(
        f"{feature}={value:+.4f}"
        for feature, value in ranked[:6]
    )


def select_case_indices(
    candidates: pd.DataFrame,
) -> list[int]:
    """
    Explicitly select both prediction classes.

    This avoids the previous ambiguity where sorting alone could yield
    five suspicious cases even though normal predictions existed.
    """

    candidates = candidates.copy()

    suspicious = candidates[
        candidates["predicted_class"] == 1
    ].sort_values(
        "predicted_suspicious_probability",
        ascending=False,
    )

    normal = candidates[
        candidates["predicted_class"] == 0
    ].sort_values(
        "predicted_suspicious_probability",
        ascending=True,
    )

    candidates["threshold_distance"] = (
        candidates[
            "predicted_suspicious_probability"
        ] - 0.50
    ).abs()

    middle = candidates.sort_values(
        "threshold_distance",
        ascending=True,
    )

    selected: list[int] = []

    # Up to two explicitly suspicious.
    for _, row in suspicious.head(2).iterrows():
        selected.append(
            int(row["external_row_index"])
        )

    # Up to two explicitly normal.
    for _, row in normal.head(2).iterrows():
        idx = int(row["external_row_index"])
        if idx not in selected:
            selected.append(idx)

    # One borderline case.
    for _, row in middle.iterrows():
        idx = int(row["external_row_index"])
        if idx not in selected:
            selected.append(idx)
            break

    # Guarantee at least three cases when enough candidates exist.
    for _, row in candidates.iterrows():
        idx = int(row["external_row_index"])
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= 5:
            break

    return selected[:5]


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    source_path = Path(args.source)
    embeddings_path = Path(args.embeddings)
    output_dir = Path(args.output_dir)
    reuse_embedding_path = Path(args.reuse_external_embeddings)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("PHASE 4 (V2) — EXTERNAL CASE STUDIES")
    print("=" * 80)
    print(f"External input:       {input_path}")
    print(f"Source dataset:       {source_path}")
    print(f"Source embeddings:    {embeddings_path}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(
        "Final source model:   "
        f"XGBoost{' + SMOTE' if args.smote else ''} trained on ALL source rows"
    )
    print("External->external:   NONE")
    print("External used in fit: NO")
    print("=" * 80)

    if not input_path.exists():
        raise FileNotFoundError(
            f"External dataset not found: {input_path}"
        )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {source_path}"
        )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Source embeddings not found: {embeddings_path}"
        )

    # --------------------------------------------------------------
    # Source
    # --------------------------------------------------------------

    source = prepare_source(
        pd.read_csv(source_path)
    )

    source_embeddings = np.load(
        embeddings_path
    ).astype(np.float32)

    print(
        f"\nSource rows: {len(source)}"
    )
    print(
        f"Source embedding shape: {source_embeddings.shape}"
    )

    if len(source) != 2011:
        raise ValueError(
            f"Expected 2,011 source rows, found {len(source)}."
        )

    if source_embeddings.shape != (2011, 384):
        raise ValueError(
            "Expected source embeddings with shape (2011, 384), "
            f"found {source_embeddings.shape}."
        )

    source_graph = build_source_graph(
        source_embeddings
    )

    print(
        f"Source graph nodes: {source_graph.number_of_nodes()}"
    )
    print(
        f"Source graph edges: {source_graph.number_of_edges()}"
    )

    graph_df, labels = source_features_and_labels(
        source_graph
    )

    source_model_df = source.copy()
    source_model_df["graph_index"] = np.arange(
        len(source_model_df)
    )

    source_model_df = source_model_df.merge(
        graph_df,
        on="graph_index",
        how="left",
    )

    source_model_df["weak_label"] = labels

    print(
        "\nSource weak labels:"
    )
    print(
        source_model_df["weak_label"].value_counts().sort_index()
    )

    # --------------------------------------------------------------
    # Final model: train on ALL 2,011 source reviews
    # --------------------------------------------------------------

    X_source = source_model_df[
        FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)

    y_source = source_model_df[
        "weak_label"
    ].astype(int)

    model = build_xgb()

    if args.smote:
        if SMOTE is None:
            raise ImportError(
                "Install imbalanced-learn before using --smote."
            )

        smote = SMOTE(
            random_state=RANDOM_STATE
        )

        X_fit, y_fit = smote.fit_resample(
            X_source,
            y_source,
        )

        print(
            f"\nSMOTE fit rows: {len(X_fit)}"
        )
    else:
        X_fit = X_source
        y_fit = y_source

    print(
        f"Training final model on {len(X_fit)} rows..."
    )

    model.fit(
        X_fit,
        y_fit,
    )

    joblib.dump(
        model,
        output_dir / "source_model_full.joblib",
    )

    # --------------------------------------------------------------
    # External dataset
    # --------------------------------------------------------------

    external = validate_external(
        pd.read_csv(input_path)
    )

    print(
        f"\nExternal raw rows: {len(pd.read_csv(input_path))}"
    )
    print(
        f"External retained rows: {len(external)}"
    )

    if len(external) == 4842:
        print(
            "PASS: exactly 4,842 retained external reviews."
        )
    else:
        print(
            "WARNING: expected 4,842 retained reviews."
        )

    external_embedding_file = reuse_embedding_path

    if external_embedding_file.exists():
        print(
            "\nReusing existing external MiniLM embeddings:"
        )
        print(
            external_embedding_file
        )

        external_embeddings = np.load(
            external_embedding_file
        ).astype(np.float32)

        if len(external_embeddings) != len(external):
            raise ValueError(
                "Saved external embeddings do not match external rows: "
                f"{len(external_embeddings)} vs {len(external)}"
            )

    else:
        print(
            "\nEncoding external reviews with MiniLM..."
        )

        encoder = SentenceTransformer(
            EMBEDDING_MODEL
        )

        external_embeddings = encoder.encode(
            external[
                "clean_review_text"
            ].tolist(),
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        np.save(
            output_dir / "external_minilm_embeddings.npy",
            external_embeddings,
        )

    print(
        f"External embedding shape: "
        f"{external_embeddings.shape}"
    )

    if external_embeddings.shape[1] != 384:
        raise ValueError(
            "Expected 384-dimensional external embeddings."
        )

    # Always save a copy with v2.
    np.save(
        output_dir / "external_minilm_embeddings.npy",
        external_embeddings,
    )

    # --------------------------------------------------------------
    # External candidate projection
    # --------------------------------------------------------------

    (
        candidate_indices,
        similarity_matrix,
        projection_degree,
        projection_max_similarity,
    ) = choose_candidates(
        external_embeddings,
        source_embeddings,
        args.candidate_pool,
    )

    print(
        f"\nExact candidate projections: "
        f"{len(candidate_indices)}"
    )

    candidate_rows = []

    for position, external_idx in enumerate(
        candidate_indices,
        start=1,
    ):
        graph_values = projection_features(
            external_embeddings[external_idx],
            source_graph,
            source_embeddings,
        )

        row = external.iloc[
            int(external_idx)
        ].to_dict()

        for feature in GRAPH_FEATURES_NO_LEAKAGE:
            row[feature] = graph_values[feature]

        row["projection_degree"] = graph_values[
            "projection_degree"
        ]

        row[
            "projection_max_similarity"
        ] = graph_values[
            "projection_max_similarity"
        ]

        row[
            "external_row_index"
        ] = int(external_idx)

        candidate_rows.append(row)

        if (
            position % 10 == 0
            or position == len(candidate_indices)
        ):
            print(
                f"  Projected {position}/"
                f"{len(candidate_indices)}"
            )

    candidates = pd.DataFrame(
        candidate_rows
    )

    candidates[FEATURES] = (
        candidates[FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0)
    )

    candidates[
        "predicted_suspicious_probability"
    ] = model.predict_proba(
        candidates[FEATURES]
    )[:, 1]

    candidates[
        "predicted_class"
    ] = (
        candidates[
            "predicted_suspicious_probability"
        ] >= 0.50
    ).astype(int)

    candidates.to_csv(
        output_dir /
        "external_projection_candidates.csv",
        index=False,
    )

    suspicious_count = int(
        candidates["predicted_class"].sum()
    )

    normal_count = int(
        len(candidates) - suspicious_count
    )

    print(
        "\nCandidate prediction distribution:"
    )
    print(
        f"  Predicted suspicious: {suspicious_count}"
    )
    print(
        f"  Predicted normal:     {normal_count}"
    )

    # --------------------------------------------------------------
    # Explicit 3-5 case selection
    # --------------------------------------------------------------

    selected_indices = select_case_indices(
        candidates
    )

    case_pool = candidates[
        candidates["external_row_index"].isin(
            selected_indices
        )
    ].copy()

    # Print selected classes BEFORE SHAP so selection is auditable.
    print(
        "\nSelected case distribution:"
    )
    print(
        case_pool[
            "predicted_class"
        ].value_counts().sort_index()
    )

    case_rows = []

    for case_id, (_, row) in enumerate(
        case_pool.iterrows(),
        start=1,
    ):
        X_row = row[
            FEATURES
        ].to_frame().T.astype(float)

        probability = float(
            row[
                "predicted_suspicious_probability"
            ]
        )

        predicted_class = int(
            row[
                "predicted_class"
            ]
        )

        prediction_label = (
            "predicted_suspicious"
            if predicted_class == 1
            else "predicted_normal"
        )

        figure_path = (
            output_dir
            / f"case_study_{case_id}_"
              f"{prediction_label}.png"
        )

        shap_text = shap_waterfall(
            model,
            X_row,
            figure_path,
            (
                f"Case Study {case_id}: "
                f"{prediction_label.replace('_', ' ').title()} | "
                f"p={probability:.3f}"
            ),
        )

        case_rows.append(
            {
                "case_id": case_id,
                "external_row_index": int(
                    row["external_row_index"]
                ),
                "hotel_name": str(
                    row.get("hotel_name", "")
                ),
                "review_title": str(
                    row.get("review_title", "")
                ),
                "prediction_label": prediction_label,
                "predicted_class": predicted_class,
                "predicted_suspicious_probability": probability,
                "projection_degree": float(
                    row["projection_degree"]
                ),
                "projection_max_similarity": float(
                    row["projection_max_similarity"]
                ),
                "review_score_numeric": float(
                    row["review_score_numeric"]
                ),
                "token_count": float(
                    row["token_count"]
                ),
                "character_count": float(
                    row["character_count"]
                ),
                "betweenness_centrality": float(
                    row["betweenness_centrality"]
                ),
                "clustering_coefficient": float(
                    row["clustering_coefficient"]
                ),
                "triangle_count": float(
                    row["triangle_count"]
                ),
                "eigenvector_centrality": float(
                    row["eigenvector_centrality"]
                ),
                "component_size": float(
                    row["component_size"]
                ),
                "top_shap_features": shap_text,
                "review_text": str(
                    row["review_text"]
                ),
                "shap_figure": str(
                    figure_path
                ),
            }
        )

    case_studies = pd.DataFrame(
        case_rows
    )

    case_studies.to_csv(
        output_dir /
        "case_studies.csv",
        index=False,
    )

    with open(
        output_dir /
        "run_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "PHASE 4 V2 — EXTERNAL CASE STUDIES\n"
        )
        f.write("=" * 60 + "\n")
        f.write(
            f"External raw rows: {len(pd.read_csv(input_path))}\n"
        )
        f.write(
            f"External retained rows: {len(external)}\n"
        )
        f.write(
            f"Source rows: {len(source)}\n"
        )
        f.write(
            f"Source graph nodes: "
            f"{source_graph.number_of_nodes()}\n"
        )
        f.write(
            f"Source graph edges: "
            f"{source_graph.number_of_edges()}\n"
        )
        f.write(
            f"Source suspicious weak labels: "
            f"{int(y_source.sum())}\n"
        )
        f.write(
            f"Projected candidate rows: "
            f"{len(candidates)}\n"
        )
        f.write(
            f"Candidate suspicious predictions: "
            f"{suspicious_count}\n"
        )
        f.write(
            f"Candidate normal predictions: "
            f"{normal_count}\n"
        )
        f.write(
            f"Case studies: "
            f"{len(case_studies)}\n"
        )
        f.write(
            "External verified labels: unavailable\n"
        )
        f.write(
            "Interpretation: model predictions only; "
            "not verified deceptive/truthful labels.\n"
        )

    print("\n" + "=" * 80)
    print("PHASE 4 V2 COMPLETE")
    print("=" * 80)

    print(
        f"External retained: {len(external)}"
    )
    print(
        f"Candidate suspicious: {suspicious_count}"
    )
    print(
        f"Candidate normal: {normal_count}"
    )
    print(
        f"Case studies selected: {len(case_studies)}"
    )

    print("\nSelected cases:")
    print(
        case_studies[
            [
                "case_id",
                "prediction_label",
                "predicted_suspicious_probability",
                "hotel_name",
                "review_title",
                "top_shap_features",
            ]
        ].to_string(index=False)
    )

    print("\nOutputs:")
    print(
        output_dir / "case_studies.csv"
    )
    print(
        output_dir / "external_projection_candidates.csv"
    )
    print(
        output_dir / "source_model_full.joblib"
    )
    print(
        output_dir / "run_summary.txt"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
