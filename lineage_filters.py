import pandas as pd
from typing import Dict, List, Any, Tuple
from utils import METRIC_CONFIG, highlight_main_path, build_query_maps


def compute_case_stats(df: pd.DataFrame, name: str, label_path: List[str], metric: str) -> Dict[str, Any]:
    """Aggregate stats (num_cases + metric value) for a subset of cases."""
    if df.empty:
        return {"subset_name": name, "label_path": " → ".join(label_path), "num_cases": 0, metric: 0}

    column = METRIC_CONFIG[metric]["column"]
    dedup = df.drop_duplicates("case:concept:name")
    return {
        "subset_name": name,
        "label_path": " → ".join(label_path),
        "num_cases": dedup["case:concept:name"].nunique(),
        metric: dedup[column].mean(),
    }


def split_subsets(
    subsets: List[Dict[str, Any]],
    query_obj,
    filter_label: str,
    step_index: int,
    query_evaluator,
    filter_cache: Dict,
) -> List[Dict[str, Any]]:
    """Split a set of subsets into filtered vs complement subsets."""
    new_subsets = []
    for subset in subsets:
        subset_df, subset_name, path, order_path = (
            subset["df"],
            subset["name"],
            subset["label_path"],
            subset.get("order_path", []),
        )

        cache_key = (subset_name, query_obj.name)
        if cache_key in filter_cache:
            df_filtered, df_complement = filter_cache[cache_key]
        else:
            df_filtered, df_complement = query_evaluator.evaluate(subset_df, query_obj)
            filter_cache[cache_key] = (df_filtered, df_complement)

        new_subsets.extend(
            [
                {
                    "df": df_filtered,
                    "name": f"{subset_name}_F{step_index+1}",
                    "label_path": path + [f"{filter_label} ✓"],
                    "order_path": order_path + [0],
                },
                {
                    "df": df_complement,
                    "name": f"{subset_name}_C{step_index+1}",
                    "label_path": path + [f"{filter_label} ✗"],
                    "order_path": order_path + [1],
                },
            ]
        )
    return new_subsets


def apply_filters(selected_sequence_df, log_view, metric: str) -> Tuple[pd.DataFrame, List[str]]:
    """Apply filters in lineage order and compute case stats per branch."""
    initial_log_name = selected_sequence_df.iloc[0]["source_log"]
    base_df = log_view.result_set_name_cache[initial_log_name]

    enrich_fn = METRIC_CONFIG[metric]["enrich_fn"]
    initial_df = enrich_fn(base_df)

    current_subsets = [
        {
            "df": initial_df,
            "name": initial_log_name,
            "label_path": ["Initial Source"],
            "order_path": [],
            "is_main_path": True,
        }
    ]
    filter_cache, main_path_leaf = {}, None

    query_map, query_expr_map = build_query_maps(log_view.query_registry)

    for i, row in selected_sequence_df.iterrows():
        query_obj = query_map.get(row["query"])
        query_expr = query_expr_map.get(row["query"], row["labels"])
        next_subsets = []

        for subset in current_subsets:
            df, path, order_path, is_main = (
                subset["df"],
                subset["label_path"],
                subset["order_path"],
                subset["is_main_path"],
            )
            if df.empty:
                continue

            cache_key = (subset["name"], query_obj.name)
            if cache_key in filter_cache:
                df_filtered, df_complement = filter_cache[cache_key]
            else:
                df_filtered, df_complement = log_view.query_evaluator.evaluate(df, query_obj)
                filter_cache[cache_key] = (df_filtered, df_complement)

            next_subsets.extend(
                [
                    {
                        "df": df_filtered,
                        "name": f"{subset['name']}_F{i+1}",
                        "label_path": path + [f"{query_expr} ✔"],
                        "order_path": order_path + [0],
                        "is_main_path": is_main,
                    },
                    {
                        "df": df_complement,
                        "name": f"{subset['name']}_C{i+1}",
                        "label_path": path + [f"{query_expr} ✘"],
                        "order_path": order_path + [1],
                        "is_main_path": False,
                    },
                ]
            )
        current_subsets = next_subsets

    for subset in current_subsets:
        if subset["is_main_path"] and not subset["df"].empty:
            main_path_leaf = subset
            break

    main_labels = {
        tuple(main_path_leaf["label_path"][:i+1]) for i in range(len(main_path_leaf["label_path"]))
    } if main_path_leaf else set()

    result_rows = []
    for subset in current_subsets:
        df, label_path = subset["df"], subset["label_path"]
        if df.empty:
            continue

        display_path = highlight_main_path(label_path, main_labels)
        stats = compute_case_stats(df, subset["name"], display_path, metric)
        row = {
            **{f"Level{i+1}": label for i, label in enumerate(display_path)},
            "num_cases": stats["num_cases"],
            metric: stats[metric],
            "order_path": subset["order_path"],
        }
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows).sort_values(by="order_path")
    return result_df, main_path_leaf["label_path"] if main_path_leaf else []

def build_case_filter_matrix(lineage_df, log_view, case_col="case:concept:name"):
    """
    Build a case × filter matrix.
    Each cell is True/False depending on whether the case passes the filter.
    Filters are evaluated once on the original source log.
    """
    query_map, query_expr_map = build_query_maps(log_view.query_registry)

    initial_log_name = lineage_df.iloc[0]["source_log"]
    base_df = log_view.result_set_name_cache[initial_log_name]

    all_cases = pd.Index(base_df[case_col].unique(), name=case_col)
    matrix = pd.DataFrame(index=all_cases)

    for _, row in lineage_df.iterrows():
        query_name = row["query"]
        query_obj = query_map[query_name]
        query_expr = query_expr_map.get(query_name, query_name)

        passed_df, _ = log_view.query_evaluator.evaluate(base_df, query_obj)
        passed_cases = set(passed_df[case_col].unique())

        matrix[query_expr] = matrix.index.isin(passed_cases)

    return matrix


def apply_filters_from_matrix(matrix, base_df, order, metric):
    """
    Build final branch rows from a precomputed case × filter matrix.
    This allows changing filter order without recomputing filters.
    """
    enrich_fn = METRIC_CONFIG[metric]["enrich_fn"]
    enriched_df = enrich_fn(base_df)

    rows = []

    for values, group in matrix.groupby(order, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)

        selected_cases = group.index
        subset_df = enriched_df[enriched_df["case:concept:name"].isin(selected_cases)]

        label_path = ["Initial Source"]
        order_path = []

        for filter_name, passed in zip(order, values):
            if passed:
                label_path.append(f"{filter_name} ✔")
                order_path.append(0)
            else:
                label_path.append(f"{filter_name} ✘")
                order_path.append(1)

        stats = compute_case_stats(
            subset_df,
            "_".join(map(str, order_path)),
            label_path,
            metric,
        )

        row = {
            **{f"Level{i+1}": label for i, label in enumerate(label_path)},
            "num_cases": stats["num_cases"],
            metric: stats[metric],
            "order_path": order_path,
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by="order_path")

def build_registry_filter_matrix(log_view, case_col="case:concept:name"):
    """
    Build a case × filter matrix for all queries in the current registry,
    not only the lineage of one selected result set.
    """
    query_map, query_expr_map = build_query_maps(log_view.query_registry)

    base_df = log_view.query_registry.get_initial_source_log()
    all_cases = pd.Index(base_df[case_col].unique(), name=case_col)

    matrix = pd.DataFrame(index=all_cases)

    for query_name, query_obj in query_map.items():
        query_expr = query_expr_map.get(query_name, query_name)

        passed_df, _ = log_view.query_evaluator.evaluate(base_df, query_obj)
        passed_cases = set(passed_df[case_col].unique())

        matrix[query_expr] = matrix.index.isin(passed_cases)

    return matrix, base_df