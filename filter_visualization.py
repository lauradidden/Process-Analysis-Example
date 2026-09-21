import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output
from utils import format_metric_value, METRIC_CONFIG, print_summary, build_query_maps
from chart_helpers import build_case_paths, get_normalized_colors, format_slice_labels
from lineage_core import get_lineage, get_sibling_subsets
from lineage_filters import (
    apply_filters,
    build_case_filter_matrix,
    apply_filters_from_matrix,
    build_registry_filter_matrix,
)
from IPython import get_ipython
import keyword

def filterbranchmap(
    log_view,
    result_set_name=None,
    metric="avg_case_duration_seconds",
    filters=None,
    source_log=None,
    case_col="case:concept:name",
    missing_metric="drop",
    allow_filter_selection=True,
    allow_reordering=True,
    allow_subset_saving=True,
):
    """
    Build a clickable and reorderable icicle visualization of filter branches.

    Parameters
    ----------
    log_view
        Object providing ``query_registry``, ``query_evaluator``, and optionally
        ``result_set_name_cache``.

    result_set_name : str, optional
        Name of an existing result set. Its lineage is used to determine the
        initial filter order when ``filters`` is not supplied.

    metric : str or dict
        Either:

        1. A standard metric key from ``METRIC_CONFIG``:

               metric="avg_case_duration_seconds"

        2. A custom metric definition:

               custom_metric = {
                   "label": "Total order value",
                   "formula": lambda log: (
                       log.groupby("case:concept:name")["order_value"].sum()
                   ),
                   "color_scheme": "Blues",
                   "aggregation": "mean",
               }

               metric=custom_metric

        The custom formula must return one numeric value per case as either
        a Series indexed by case ID or a DataFrame containing ``case_col`` and
        ``metric_value``.

    filters : sequence, optional
        Initial filters. Entries can be registry filter names, displayed query
        expressions, or query objects.

        If omitted, the function attempts to extract the filters from the
        lineage of ``result_set_name``.

    source_log : pandas.DataFrame, optional
        Explicit original event log. If omitted, it is obtained from the result
        lineage or the query registry.

    case_col : str
        Column containing the case identifier.

    missing_metric : {"drop", "error", "zero"}
        Handling of cases without a metric:

        - ``"drop"``: cases remain in counts but do not affect node colors.
        - ``"error"``: raise an exception.
        - ``"zero"``: fill missing metrics with zero.

    allow_filter_selection : bool
        Show controls for adding and removing filters.

    allow_reordering : bool
        Show controls for moving filters up and down.

    allow_subset_saving : bool
        Show controls for saving a clicked subset.

    Returns
    -------
    dict
        A dictionary populated when subsets are saved:

            {
                "subset_name": {
                    "data": event_dataframe,
                    "case_ids": [...],
                    "path": "...",
                    "filter_order": [...],
                    "metric": {...},
                }
            }

        In a notebook, saved subsets are also inserted into the user namespace.
    """

    # ------------------------------------------------------------------
    # Validation and source-log resolution
    # ------------------------------------------------------------------

    if missing_metric not in {"drop", "error", "zero"}:
        raise ValueError(
            "missing_metric must be 'drop', 'error', or 'zero'."
        )

    query_map, query_expr_map = build_query_maps(log_view.query_registry)

    if not query_map:
        raise ValueError("No filters were found in the query registry.")

    if source_log is not None:
        base_df = source_log.copy()

    elif result_set_name is not None:
        lineage = get_lineage(
            log_view.query_registry.summary(),
            result_set_name,
        )

        if lineage.empty:
            raise ValueError(
                f"No lineage was found for result set '{result_set_name}'."
            )

        initial_log_name = lineage.iloc[0].get("source_log")

        if (
            initial_log_name is not None
            and hasattr(log_view, "result_set_name_cache")
            and initial_log_name in log_view.result_set_name_cache
        ):
            base_df = log_view.result_set_name_cache[
                initial_log_name
            ].copy()
        else:
            base_df = (
                log_view.query_registry
                .get_initial_source_log()
                .copy()
            )

    else:
        lineage = None
        base_df = (
            log_view.query_registry
            .get_initial_source_log()
            .copy()
        )

    if case_col not in base_df.columns:
        raise KeyError(
            f"The case column '{case_col}' is not present in the source log."
        )

    if base_df.empty:
        raise ValueError("The source event log is empty.")

    all_source_case_ids = pd.Index(
        base_df[case_col].dropna().unique(),
        name=case_col,
    )

    if all_source_case_ids.empty:
        raise ValueError("The source event log contains no valid case IDs.")

    # ------------------------------------------------------------------
    # Query lookup tables
    # ------------------------------------------------------------------

    # query_map is expected to map a registry name to a query object.
    query_name_by_object_id = {
        id(query_obj): query_name
        for query_name, query_obj in query_map.items()
    }

    expression_to_name = {
        expression: query_name
        for query_name, expression in query_expr_map.items()
        if query_name in query_map
    }

    def resolve_filter(filter_reference):
        """
        Resolve a filter name, expression, or query object.

        Returns
        -------
        tuple
            (registry_name, query_object)
        """

        if isinstance(filter_reference, str):
            if filter_reference in query_map:
                return (
                    filter_reference,
                    query_map[filter_reference],
                )

            if filter_reference in expression_to_name:
                query_name = expression_to_name[filter_reference]
                return query_name, query_map[query_name]

            raise KeyError(
                f"Unknown filter '{filter_reference}'. It is neither a "
                "registry filter name nor a registered query expression."
            )

        object_name = query_name_by_object_id.get(id(filter_reference))

        if object_name is not None:
            return object_name, filter_reference

        # Permit a supplied query object that is not registered.
        if hasattr(filter_reference, "as_string"):
            expression = filter_reference.as_string()
            generated_name = expression

            if generated_name not in query_map:
                query_map[generated_name] = filter_reference
                query_expr_map[generated_name] = expression
                expression_to_name[expression] = generated_name

            return generated_name, filter_reference

        raise TypeError(
            "Each filter must be a registry name, registered expression, "
            "or query object."
        )

    # ------------------------------------------------------------------
    # Determine the initial filter order
    # ------------------------------------------------------------------

    def extract_filters_from_lineage(lineage_df):
        """
        Recover registered filters from the lineage in their original order.
        """

        if lineage_df is None or lineage_df.empty:
            return []

        resolved_names = []

        for _, row in lineage_df.iterrows():
            row_values = []

            for value in row.tolist():
                # Avoid calling pd.isna() as a Boolean condition on
                # arrays, lists, dictionaries, or other containers.
                if pd.api.types.is_scalar(value) and pd.isna(value):
                    continue

                text = str(value).strip()

                if text:
                    row_values.append(text)

            matched_name = None

            # First try exact registry-name or expression matches.
            for query_name in query_map:
                expression = query_expr_map.get(query_name)

                if query_name in row_values:
                    matched_name = query_name
                    break

                if expression and expression in row_values:
                    matched_name = query_name
                    break

            # Fall back to searching within the complete row text.
            if matched_name is None:
                combined_text = " ".join(row_values)

                for query_name in query_map:
                    expression = query_expr_map.get(query_name)

                    if expression and expression in combined_text:
                        matched_name = query_name
                        break

            if (
                matched_name is not None
                and matched_name not in resolved_names
            ):
                resolved_names.append(matched_name)

        return resolved_names

    if filters is not None:
        initial_filter_names = [
            resolve_filter(filter_reference)[0]
            for filter_reference in filters
        ]
    else:
        if result_set_name is not None:
            if "lineage" not in locals():
                lineage = get_lineage(
                    log_view.query_registry.summary(),
                    result_set_name,
                )

            initial_filter_names = extract_filters_from_lineage(lineage)
        else:
            initial_filter_names = []

    # Remove accidental duplicates while preserving order.
    initial_filter_names = list(dict.fromkeys(initial_filter_names))

    available_filter_names = list(query_map.keys())

    # ------------------------------------------------------------------
    # Metric normalization
    # ------------------------------------------------------------------

    def normalize_metric_result(metric_result):
        """
        Normalize a custom metric to:

            case_col | metric_value
        """

        if isinstance(metric_result, pd.Series):
            result = (
                metric_result
                .rename("metric_value")
                .rename_axis(case_col)
                .reset_index()
            )

        elif isinstance(metric_result, pd.DataFrame):
            result = metric_result.copy()

            if case_col not in result.columns:
                raise ValueError(
                    "A custom metric DataFrame must contain the case "
                    f"column '{case_col}'."
                )

            if "metric_value" not in result.columns:
                value_columns = [
                    column
                    for column in result.columns
                    if column != case_col
                ]

                if len(value_columns) != 1:
                    raise ValueError(
                        "A custom metric DataFrame must contain a "
                        "'metric_value' column, or exactly one value "
                        "column in addition to the case column."
                    )

                result = result.rename(
                    columns={value_columns[0]: "metric_value"}
                )

            result = result[[case_col, "metric_value"]]

        elif isinstance(metric_result, dict):
            result = pd.DataFrame(
                {
                    case_col: list(metric_result.keys()),
                    "metric_value": list(metric_result.values()),
                }
            )

        else:
            raise TypeError(
                "The custom metric formula must return a pandas Series, "
                "a pandas DataFrame, or a mapping from case ID to value."
            )

        if result[case_col].isna().any():
            result = result.loc[result[case_col].notna()].copy()

        duplicate_mask = result[case_col].duplicated(keep=False)

        if duplicate_mask.any():
            duplicates = (
                result.loc[duplicate_mask, case_col]
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )

            raise ValueError(
                "The metric must produce no more than one value per case. "
                f"Duplicate case IDs include: {duplicates}"
            )

        result["metric_value"] = pd.to_numeric(
            result["metric_value"],
            errors="coerce",
        )

        return result

    def build_standard_metric(metric_name):
        if metric_name not in METRIC_CONFIG:
            available = ", ".join(METRIC_CONFIG.keys())

            raise KeyError(
                f"Unknown standard metric '{metric_name}'. "
                f"Available metrics are: {available}"
            )

        config = METRIC_CONFIG[metric_name]
        enriched_log = config["enrich_fn"](base_df.copy())

        value_col = config["column"]

        if value_col not in enriched_log.columns:
            raise KeyError(
                f"The enrichment function for '{metric_name}' did not "
                f"produce the expected column '{value_col}'."
            )

        # A valid case-level metric should be constant across all event rows
        # belonging to a case.
        values_per_case = (
            enriched_log
            .groupby(case_col, dropna=False)[value_col]
            .nunique(dropna=True)
        )

        inconsistent_cases = values_per_case[
            values_per_case > 1
        ]

        if not inconsistent_cases.empty:
            examples = (
                inconsistent_cases.index
                .astype(str)
                .tolist()[:5]
            )

            raise ValueError(
                f"Standard metric '{metric_name}' produced multiple "
                "different values within the same case. The metric must "
                "have exactly one value per case. Example case IDs: "
                f"{examples}"
            )

        case_metric = (
            enriched_log[[case_col, value_col]]
            .drop_duplicates(subset=[case_col])
            .rename(columns={value_col: "metric_value"})
        )

        case_metric = normalize_metric_result(case_metric)

        return case_metric, {
            "name": metric_name,
            "label": config["label"],
            "color_scheme": config["color_scheme"],
            "aggregation": config.get("aggregation", "mean"),
            "standard": True,
        }

    def build_custom_metric(metric_definition):
        if "label" not in metric_definition:
            raise KeyError(
                "A custom metric definition must contain 'label'."
            )

        if "formula" not in metric_definition:
            raise KeyError(
                "A custom metric definition must contain 'formula'."
            )

        formula = metric_definition["formula"]

        if not callable(formula):
            raise TypeError(
                "The custom metric 'formula' must be callable."
            )

        metric_result = formula(base_df.copy())
        case_metric = normalize_metric_result(metric_result)

        return case_metric, {
            "name": metric_definition.get(
                "name",
                metric_definition["label"],
            ),
            "label": metric_definition["label"],
            "color_scheme": metric_definition.get(
                "color_scheme",
                "Viridis",
            ),
            "aggregation": metric_definition.get(
                "aggregation",
                "mean",
            ),
            "standard": False,
        }

    if isinstance(metric, str):
        case_metric_df, metric_info = build_standard_metric(metric)

    elif isinstance(metric, dict):
        case_metric_df, metric_info = build_custom_metric(metric)

    else:
        raise TypeError(
            "metric must be either a standard metric name or a custom "
            "metric definition dictionary."
        )

    supported_aggregations = {
        "mean",
        "median",
        "sum",
        "min",
        "max",
    }

    if metric_info["aggregation"] not in supported_aggregations:
        raise ValueError(
            f"Unsupported metric aggregation "
            f"'{metric_info['aggregation']}'. Supported values are: "
            f"{sorted(supported_aggregations)}"
        )

    source_case_set = set(all_source_case_ids)

    unknown_metric_cases = set(
        case_metric_df[case_col]
    ) - source_case_set

    if unknown_metric_cases:
        examples = list(unknown_metric_cases)[:5]

        raise ValueError(
            "The metric contains case IDs that are not present in the "
            f"source log. Examples: {examples}"
        )

    all_cases_df = pd.DataFrame(
        {case_col: all_source_case_ids}
    )

    all_cases_df = all_cases_df.merge(
        case_metric_df,
        on=case_col,
        how="left",
        validate="one_to_one",
    )

    missing_count = int(
        all_cases_df["metric_value"].isna().sum()
    )

    if missing_count and missing_metric == "error":
        raise ValueError(
            f"The metric is missing for {missing_count:,} source cases."
        )

    if missing_metric == "zero":
        all_cases_df["metric_value"] = (
            all_cases_df["metric_value"].fillna(0.0)
        )

    # ------------------------------------------------------------------
    # Build the independent case-filter matrix
    # ------------------------------------------------------------------

    filter_case_sets = {}

    for filter_name in available_filter_names:
        query_obj = query_map[filter_name]

        passed_df, _ = log_view.query_evaluator.evaluate(
            base_df.copy(),
            query_obj,
        )

        if case_col not in passed_df.columns:
            raise KeyError(
                f"Filter '{filter_name}' returned data without the "
                f"case column '{case_col}'."
            )

        filter_case_sets[filter_name] = set(
            passed_df[case_col].dropna().unique()
        )

    case_matrix = all_cases_df.copy()

    for filter_name in available_filter_names:
        case_matrix[filter_name] = (
            case_matrix[case_col]
            .isin(filter_case_sets[filter_name])
        )

    # ------------------------------------------------------------------
    # Widget state
    # ------------------------------------------------------------------

    selected_sublogs = {}

    pending_selection = {
        "data": None,
        "case_ids": None,
        "path": None,
        "node_id": None,
        "filter_order": None,
    }

    node_case_ids = {}
    node_paths = {}

    available_widget = widgets.SelectMultiple(
        options=available_filter_names,
        value=(),
        description="Available",
        rows=max(5, min(len(available_filter_names), 15)),
        layout=widgets.Layout(width="650px"),
    )

    order_widget = widgets.Select(
        options=initial_filter_names,
        value=(
            initial_filter_names[0]
            if initial_filter_names
            else None
        ),
        description="Selected",
        rows=max(
            5,
            min(
                max(len(initial_filter_names), 1),
                15,
            ),
        ),
        layout=widgets.Layout(width="650px"),
    )

    add_filter_button = widgets.Button(
        description="Add filters",
        button_style="info",
    )

    remove_filter_button = widgets.Button(
        description="Remove filter",
        button_style="warning",
    )

    up_button = widgets.Button(description="Move up")
    down_button = widgets.Button(description="Move down")

    refresh_button = widgets.Button(
        description="Update chart",
        button_style="success",
        icon="refresh",
    )

    name_input = widgets.Text(
        value="",
        placeholder="Example: high_value_cases",
        description="Name:",
        layout=widgets.Layout(width="500px"),
    )

    save_button = widgets.Button(
        description="Save subset",
        button_style="success",
        disabled=True,
    )

    message_out = widgets.Output()
    preview_out = widgets.Output()
    chart_container = widgets.VBox([])

    def current_order():
        return list(order_widget.options)

    # ------------------------------------------------------------------
    # Build the node table for the current filter order
    # ------------------------------------------------------------------

    def aggregate_metric(values):
        valid_values = values.dropna()

        if valid_values.empty:
            return float("nan")

        aggregation = metric_info["aggregation"]

        if aggregation == "mean":
            return valid_values.mean()
        if aggregation == "median":
            return valid_values.median()
        if aggregation == "sum":
            return valid_values.sum()
        if aggregation == "min":
            return valid_values.min()
        if aggregation == "max":
            return valid_values.max()

        raise ValueError(
            f"Unsupported aggregation '{aggregation}'."
        )

    def build_node_table(filter_order):
        """
        Create one row per icicle node and retain a node-to-case mapping.
        """

        node_case_ids.clear()
        node_paths.clear()

        records = []
        node_counter = 0

        def add_node(
            parent_id,
            label,
            path_labels,
            subset,
            depth,
        ):
            nonlocal node_counter

            node_id = f"node_{node_counter}"
            node_counter += 1

            case_ids = subset[case_col].tolist()
            metric_values = subset["metric_value"]

            node_case_ids[node_id] = set(case_ids)
            node_paths[node_id] = list(path_labels)

            records.append(
                {
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "label": label,
                    "depth": depth,
                    "num_cases": int(subset[case_col].nunique()),
                    "num_metric_cases": int(
                        metric_values.notna().sum()
                    ),
                    "metric_value": aggregate_metric(
                        metric_values
                    ),
                    "path": " → ".join(path_labels),
                }
            )

            return node_id

        root_id = add_node(
            parent_id="",
            label="Initial Source",
            path_labels=["Initial Source"],
            subset=case_matrix,
            depth=0,
        )

        def add_children(
            parent_id,
            subset,
            filter_position,
            path_labels,
        ):
            if filter_position >= len(filter_order):
                return

            filter_name = filter_order[filter_position]
            expression = query_expr_map.get(
                filter_name,
                filter_name,
            )

            for passed, symbol in ((True, "✔"), (False, "✘")):
                branch_subset = subset.loc[
                    subset[filter_name] == passed
                ].copy()

                if branch_subset.empty:
                    continue

                branch_label = f"{expression} {symbol}"
                branch_path = path_labels + [branch_label]

                child_id = add_node(
                    parent_id=parent_id,
                    label=branch_label,
                    path_labels=branch_path,
                    subset=branch_subset,
                    depth=filter_position + 1,
                )

                add_children(
                    parent_id=child_id,
                    subset=branch_subset,
                    filter_position=filter_position + 1,
                    path_labels=branch_path,
                )

        add_children(
            parent_id=root_id,
            subset=case_matrix,
            filter_position=0,
            path_labels=["Initial Source"],
        )

        node_df = pd.DataFrame(records)

        return node_df

    # ------------------------------------------------------------------
    # Chart creation and click handling
    # ------------------------------------------------------------------

    def format_node_metric(value):
        if pd.isna(value):
            return "No metric values"

        if metric_info["standard"]:
            try:
                return format_metric_value(
                    metric_info["name"],
                    value,
                )
            except Exception:
                return f"{value:,.2f}"

        return f"{value:,.2f}"

    def render_chart():
        filter_order = current_order()

        if not filter_order:
            with message_out:
                clear_output(wait=True)
                print(
                    "Add at least one filter before building the chart."
                )

            chart_container.children = tuple()
            return

        node_df = build_node_table(filter_order)

        node_df["hover_text"] = node_df.apply(
            lambda row: (
                f"<b>{row['path']}</b><br>"
                f"{int(row['num_cases']):,} cases<br>"
                f"Cases with metric: "
                f"{int(row['num_metric_cases']):,}<br>"
                f"{metric_info['label']}: "
                f"{format_node_metric(row['metric_value'])}"
            ),
            axis=1,
        )

        fig = go.FigureWidget(
            go.Icicle(
                ids=node_df["node_id"],
                labels=node_df["label"],
                parents=node_df["parent_id"],
                values=node_df["num_cases"],
                branchvalues="total",
                customdata=node_df[
                    [
                        "hover_text",
                        "node_id",
                        "path",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "%{customdata[0]}"
                    "<br><i>Click to preview this subset</i>"
                    "<extra></extra>"
                ),
                marker=dict(
                    colors=node_df["metric_value"],
                    colorscale=metric_info["color_scheme"],
                    colorbar=dict(
                        title=metric_info["label"],
                    ),
                ),
                root=dict(color="lightgrey"),
                tiling=dict(
                    orientation="h",
                    pad=2,
                ),
            )
        )

        fig.update_layout(
            title=(
                f"Filter Branch Map"
                + (
                    f": {result_set_name}"
                    if result_set_name
                    else ""
                )
                + "<br>"
                + f"Metric: {metric_info['label']}"
            ),
            margin=dict(t=90, l=20, r=20, b=20),
            width=1200,
            height=700,
        )

        def on_chart_click(trace, points, selector):
            if not points.point_inds:
                return

            point_index = points.point_inds[0]
            node_id = str(trace.ids[point_index])

            selected_case_ids = node_case_ids.get(
                node_id,
                set(),
            )

            subset_df = base_df.loc[
                base_df[case_col].isin(selected_case_ids)
            ].copy()

            clicked_path = " → ".join(
                node_paths.get(node_id, [])
            )

            pending_selection["data"] = subset_df
            pending_selection["case_ids"] = set(
                selected_case_ids
            )
            pending_selection["path"] = clicked_path
            pending_selection["node_id"] = node_id
            pending_selection["filter_order"] = list(
                filter_order
            )

            save_button.disabled = not allow_subset_saving

            with preview_out:
                clear_output(wait=True)

                print(f"Selected node: {clicked_path}")
                print(
                    f"{len(selected_case_ids):,} cases and "
                    f"{len(subset_df):,} events."
                )

                if allow_subset_saving:
                    print(
                        "Enter a name and click 'Save subset' "
                        "to save these original event rows."
                    )

                display(subset_df.head(20))

        fig.data[0].on_click(on_chart_click)

        chart_container.children = (fig,)

        pending_selection.update(
            {
                "data": None,
                "case_ids": None,
                "path": None,
                "node_id": None,
                "filter_order": None,
            }
        )

        save_button.disabled = True

        with preview_out:
            clear_output(wait=True)
            print("Click an icicle node to preview its event data.")

        with message_out:
            clear_output(wait=True)
            print(
                f"Chart updated with {len(filter_order)} filters: "
                f"{' → '.join(filter_order)}"
            )

            if missing_count:
                print(
                    f"{missing_count:,} cases have no metric value. "
                    "They remain in case counts but do not contribute "
                    "to metric aggregation."
                )

    # ------------------------------------------------------------------
    # Filter-selection callbacks
    # ------------------------------------------------------------------

    def on_add_filters(_):
        selected = current_order()

        for filter_name in available_widget.value:
            if filter_name not in selected:
                selected.append(filter_name)

        order_widget.options = selected
        order_widget.value = (
            selected[0] if selected else None
        )

        with message_out:
            clear_output(wait=True)

            if selected:
                print(
                    "Filters added. Reorder them if needed, then "
                    "click 'Update chart'."
                )
            else:
                print("Select one or more available filters first.")

    def on_remove_filter(_):
        selected_filter = order_widget.value

        if selected_filter is None:
            with message_out:
                clear_output(wait=True)
                print("Select a filter to remove.")
            return

        updated = [
            filter_name
            for filter_name in current_order()
            if filter_name != selected_filter
        ]

        order_widget.options = updated
        order_widget.value = (
            updated[0] if updated else None
        )

        with message_out:
            clear_output(wait=True)
            print(f"Removed filter: {selected_filter}")

    def move_filter(delta):
        selected_filter = order_widget.value
        order = current_order()

        if selected_filter is None:
            with message_out:
                clear_output(wait=True)
                print("Select a filter to move.")
            return

        current_index = order.index(selected_filter)
        new_index = current_index + delta

        if new_index < 0:
            with message_out:
                clear_output(wait=True)
                print(
                    f"'{selected_filter}' is already at the top."
                )
            return

        if new_index >= len(order):
            with message_out:
                clear_output(wait=True)
                print(
                    f"'{selected_filter}' is already at the bottom."
                )
            return

        order[current_index], order[new_index] = (
            order[new_index],
            order[current_index],
        )

        order_widget.options = order
        order_widget.value = selected_filter

    def on_move_up(_):
        move_filter(-1)

    def on_move_down(_):
        move_filter(1)

    def on_refresh(_):
        render_chart()

    # ------------------------------------------------------------------
    # Subset-saving callback
    # ------------------------------------------------------------------

    def on_save_subset(_):
        subset_df = pending_selection["data"]

        if subset_df is None:
            with preview_out:
                clear_output(wait=True)
                print("Click an icicle node before saving.")
            return

        subset_name = name_input.value.strip()

        if not subset_name:
            with preview_out:
                clear_output(wait=True)
                print("Enter a name for the selected subset.")
            return

        if (
            not subset_name.isidentifier()
            or keyword.iskeyword(subset_name)
        ):
            with preview_out:
                clear_output(wait=True)
                print(
                    f"'{subset_name}' is not a valid Python "
                    "variable name."
                )
                print(
                    "Use letters, numbers, and underscores, and do "
                    "not start the name with a number."
                )
            return

        ip = get_ipython()

        if ip is not None and subset_name in ip.user_ns:
            with preview_out:
                clear_output(wait=True)
                print(
                    f"'{subset_name}' already exists in the notebook."
                )
                print(
                    "Choose another name to avoid overwriting it."
                )
            return

        saved_df = subset_df.copy()

        selected_sublogs[subset_name] = {
            "data": saved_df,
            "case_ids": sorted(
                pending_selection["case_ids"],
                key=str,
            ),
            "path": pending_selection["path"],
            "node_id": pending_selection["node_id"],
            "filter_order": list(
                pending_selection["filter_order"]
            ),
            "metric": dict(metric_info),
            "name": subset_name,
        }

        if ip is not None:
            ip.user_ns[subset_name] = saved_df

        with preview_out:
            clear_output(wait=True)

            print("Subset saved.")
            print(f"Name: {subset_name}")
            print(f"Path: {pending_selection['path']}")
            print(
                f"{saved_df[case_col].nunique():,} cases and "
                f"{len(saved_df):,} events."
            )
            print()
            print("Access the event log directly with:")
            print(subset_name)
            print()
            print("Or through the returned dictionary with:")
            print(
                f"selected_sublogs['{subset_name}']['data']"
            )

            display(saved_df.head(20))

    # ------------------------------------------------------------------
    # Connect callbacks
    # ------------------------------------------------------------------

    add_filter_button.on_click(on_add_filters)
    remove_filter_button.on_click(on_remove_filter)
    up_button.on_click(on_move_up)
    down_button.on_click(on_move_down)
    refresh_button.on_click(on_refresh)
    save_button.on_click(on_save_subset)

    # ------------------------------------------------------------------
    # Assemble the interface
    # ------------------------------------------------------------------

    controls = [
        widgets.HTML("<h3>Filter Branch Map</h3>"),
        widgets.HTML(
            f"<b>Metric:</b> {metric_info['label']}<br>"
            f"<b>Node aggregation:</b> "
            f"{metric_info['aggregation']}"
        ),
    ]

    if allow_filter_selection:
        controls.extend(
            [
                widgets.HTML(
                    "<h4>1. Select filters</h4>"
                    "<p>Select registry filters to include in the map.</p>"
                ),
                available_widget,
                add_filter_button,
            ]
        )

    controls.extend(
        [
            widgets.HTML(
                "<h4>2. Filter order</h4>"
                "<p>The order below determines the icicle hierarchy. "
                "Each filter is evaluated independently against the "
                "original source log.</p>"
            ),
            order_widget,
        ]
    )

    order_buttons = []

    if allow_reordering:
        order_buttons.extend([up_button, down_button])

    if allow_filter_selection:
        order_buttons.append(remove_filter_button)

    if order_buttons:
        controls.append(widgets.HBox(order_buttons))

    controls.extend(
        [
            refresh_button,
            message_out,
            widgets.HTML("<h4>Filter branch visualization</h4>"),
            chart_container,
            widgets.HTML("<h4>Clicked subset preview</h4>"),
            preview_out,
        ]
    )

    if allow_subset_saving:
        controls.extend(
            [
                widgets.HTML(
                    "<p>After clicking a node, enter a valid Python "
                    "variable name and save the corresponding original "
                    "event rows.</p>"
                ),
                widgets.HBox([name_input, save_button]),
            ]
        )

    ui = widgets.VBox(
        controls,
        layout=widgets.Layout(
            width="100%",
            align_items="stretch",
        ),
    )

    display(ui)

    if initial_filter_names:
        render_chart()
    else:
        with message_out:
            print(
                "Select one or more filters and click 'Update chart'."
            )

    return selected_sublogs


def query_exploration_icicle(
    result_set_name, log_view, metric="avg_case_duration_seconds", details=True
):
    """Build an icicle chart for query exploration."""

    lineage = get_lineage(log_view.query_registry.summary(), result_set_name)
    icicle_df, main_path = apply_filters(lineage, log_view, metric)

    path_cols = [c for c in icicle_df.columns if c.startswith("Level")]

    icicle_df["hover_text"] = icicle_df.apply(
        lambda row: (
            f"<b>{int(row['num_cases']):,} cases</b><br>"
            f"{METRIC_CONFIG[metric]['label']}: {format_metric_value(metric, row[metric])}"
        ),
        axis=1,
    )

    fig = px.icicle(
        icicle_df,
        path=path_cols,
        values="num_cases",
        color=metric,
        custom_data=["hover_text"],
        color_continuous_scale=METRIC_CONFIG[metric]["color_scheme"],
        title=f"Icicle Chart for: {result_set_name}",
    )

    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        coloraxis_colorbar=dict(title=METRIC_CONFIG[metric]["label"]),
    )
    fig.show()

    if details:
        print("\nSummary with Metrics:\n")
        final_path = " → ".join(main_path) if main_path else ""
        print_summary(icicle_df, path_cols, final_path, metric, METRIC_CONFIG[metric]["label"])


def query_breakdown_pie(result_set_name, log_view, metric="avg_case_duration_seconds", details=True):
    """Build a pie chart showing breakdown of cases along filter paths."""

    parent_log, query_obj, label, step_index, lineage_df = get_sibling_subsets(
        result_set_name, log_view
    )

    config = METRIC_CONFIG[metric]
    full_log = log_view.query_registry.get_initial_source_log()
    full_log = config["enrich_fn"](full_log)
    value_col, color_scheme, color_title = (
        config["column"],
        config["color_scheme"],
        config["label"],
    )

    filtered, _ = log_view.query_evaluator.evaluate(full_log, query_obj)
    if filtered.empty:
        print("No cases passed the final filter, pie chart cannot be built.")
        return

    case_paths, final_result_path = build_case_paths(lineage_df, full_log, filtered, log_view)

    filtered = filtered.copy()
    filtered["path_label"] = filtered["case:concept:name"].map(
        lambda cid: " → ".join(case_paths.get(cid, []))
    )

    grouped = (
        filtered.groupby("path_label")[["case:concept:name", value_col]]
        .agg(num_cases=("case:concept:name", "nunique"), avg_metric=(value_col, "mean"))
        .reset_index()
    )
    grouped = grouped.copy()
    grouped["wrapped_path"] = grouped["path_label"].str.replace(" → ", " →<br>")
    grouped = grouped.rename(columns={"avg_metric": metric})

    color_values = get_normalized_colors(grouped[metric], color_scheme)
    grouped = format_slice_labels(grouped, final_result_path)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=grouped["slice_label"],
                values=grouped["num_cases"],
                textinfo="label",
                customdata=grouped[["wrapped_path"]],
                hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
                marker=dict(colors=color_values),
            )
        ],
        layout=go.Layout(
            title=dict(text=f"Breakdown of Filter: {query_obj.as_string()}", x=0.5),
            width=800,
            height=700,
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=color_scheme,
                cmin=grouped[metric].min(),
                cmax=grouped[metric].max(),
                colorbar=dict(title=color_title, len=0.8, thickness=15),
                color=[grouped[metric].min()],
                showscale=True,
            ),
            hoverinfo="none",
            showlegend=False,
        )
    )

    fig.show()

    if details:
        print("\nFilter Paths:\n")
        print_summary(grouped, ["path_label"], final_result_path, metric, color_title)


def query_exploration_icicle_clickable(
    result_set_name,
    log_view,
    metric="avg_case_duration_seconds",
    details=True,
    case_col="case:concept:name",
):
    """
    Build an icicle chart where clicking a node previews that subset.
    The analyst can then enter a name and save the subset as a notebook variable.

    Usage:
        selected_sublogs = query_exploration_icicle_clickable(...)

    After clicking a node and saving it with the name 'my_sublog':

        my_sublog

    or:

        selected_sublogs["my_sublog"]["data"]
    """

    lineage = get_lineage(log_view.query_registry.summary(), result_set_name)

    initial_log_name = lineage.iloc[0]["source_log"]
    base_df = log_view.result_set_name_cache[initial_log_name]

    icicle_df, main_path = apply_filters(lineage, log_view, metric)
    path_cols = [c for c in icicle_df.columns if c.startswith("Level")]

    query_map, query_expr_map = build_query_maps(log_view.query_registry)
    expr_to_query = {
        expr: query_map[query_name]
        for query_name, expr in query_expr_map.items()
        if query_name in query_map
    }

    icicle_df["hover_text"] = icicle_df.apply(
        lambda row: (
            f"<b>{int(row['num_cases']):,} cases</b><br>"
            f"{METRIC_CONFIG[metric]['label']}: {format_metric_value(metric, row[metric])}<br>"
            f"<br><i>Click to preview this subset</i>"
        ),
        axis=1,
    )

    fig_px = px.icicle(
        icicle_df,
        path=path_cols,
        values="num_cases",
        color=metric,
        custom_data=["hover_text"],
        color_continuous_scale=METRIC_CONFIG[metric]["color_scheme"],
        title=f"Clickable Icicle Chart for: {result_set_name}",
    )

    fig = go.FigureWidget(fig_px)

    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(
        margin=dict(t=60, l=20, r=20, b=20),
        width=1200,
        height=650,
        coloraxis_colorbar=dict(title=METRIC_CONFIG[metric]["label"]),
    )

    selected_sublogs = {}
    pending_selection = {
        "data": None,
        "path": None,
    }

    name_input = widgets.Text(
        value="",
        placeholder="Enter sublog name, e.g. sublog_high_amount",
        description="Name:",
    )

    save_button = widgets.Button(
        description="Save this subset",
        button_style="success",
        disabled=True,
    )

    out = widgets.Output()

    def get_clicked_path(trace, point_index):
        ids = list(trace.ids)
        labels = list(trace.labels)
        parents = list(trace.parents)

        current_id = ids[point_index]
        id_to_label = dict(zip(ids, labels))
        id_to_parent = dict(zip(ids, parents))

        path = []

        while current_id:
            label = id_to_label.get(current_id)
            if label:
                path.append(str(label).replace("🟡 ", "").strip())
            current_id = id_to_parent.get(current_id)

        return list(reversed(path))

    def subset_from_path(path):
        selected_df = base_df.copy()

        for part in path:
            clean_part = str(part).replace("🟡 ", "").strip()

            if clean_part == "Initial Source":
                continue

            if clean_part.endswith("✔"):
                query_expr = clean_part[:-1].strip()
                passed = True
            elif clean_part.endswith("✘"):
                query_expr = clean_part[:-1].strip()
                passed = False
            else:
                continue

            query_obj = expr_to_query.get(query_expr)

            if query_obj is None:
                continue

            passed_df, complement_df = log_view.query_evaluator.evaluate(
                selected_df,
                query_obj,
            )

            selected_df = passed_df if passed else complement_df

        return selected_df

    def on_click(trace, points, selector):
        if not points.point_inds:
            return

        point_index = points.point_inds[0]
        clicked_path_parts = get_clicked_path(trace, point_index)
        clicked_path = " → ".join(clicked_path_parts)

        subset_df = subset_from_path(clicked_path_parts)

        pending_selection["data"] = subset_df
        pending_selection["path"] = clicked_path

        save_button.disabled = False

        num_cases = (
            subset_df[case_col].nunique()
            if case_col in subset_df.columns
            else len(subset_df)
        )

        with out:
            clear_output(wait=True)
            print(f"Selected node: {clicked_path}")
            print(f"This subset contains {num_cases:,} cases and {len(subset_df):,} events.")
            print("Enter a name and click 'Save this subset' to save it as an accessible sublog.")
            display(subset_df.head(20))

    def on_save_subset(_):
        if pending_selection["data"] is None:
            with out:
                clear_output(wait=True)
                print("Please click a node first.")
            return

        sublog_name = name_input.value.strip()

        if not sublog_name:
            with out:
                clear_output(wait=True)
                print("Please enter a name for this sublog.")
            return

        if not sublog_name.isidentifier() or keyword.iskeyword(sublog_name):
            with out:
                clear_output(wait=True)
                print(f"'{sublog_name}' is not a valid Python variable name.")
                print()
                print("A valid name:")
                print("• starts with a letter or '_'")
                print("• contains only letters, numbers and '_'")
                print("• cannot be a Python keyword (e.g. 'for', 'class')")
                print()
                print("Example: sublog_high_amount")
            return

        ip = get_ipython()

        if ip is not None and sublog_name in ip.user_ns:
            with out:
                clear_output(wait=True)
                print(f"'{sublog_name}' already exists in the notebook.")
                print("Please choose a different name to avoid overwriting it.")
            return

        sublog_df = pending_selection["data"]

        selected_sublogs[sublog_name] = {
            "data": sublog_df,
            "path": pending_selection["path"],
            "name": sublog_name,
        }

        if ip is not None:
            ip.user_ns[sublog_name] = sublog_df

        with out:
            clear_output(wait=True)
            print("Subset saved.")
            print()
            print(f"Name: {sublog_name}")
            print()
            print("You can access it directly with:")
            print(sublog_name)
            print()
            print("Or from the returned dictionary with:")
            print(f"selected_sublogs['{sublog_name}']['data']")
            print()
            print(f"Selected node: {pending_selection['path']}")
            display(sublog_df.head(20))

    fig.data[0].on_click(on_click)
    save_button.on_click(on_save_subset)

    display(widgets.VBox([
        fig,
        widgets.HTML("<b>Clicked subset preview</b>"),
        widgets.HTML(
            "After clicking a node, enter a name and click "
            "<b>Save this subset</b>. The name must be a valid Python variable name."
        ),
        widgets.HBox([name_input, save_button]),
        out,
    ]))

    if details:
        print("\nSummary with Metrics:\n")
        final_path = " → ".join(main_path) if main_path else ""
        print_summary(
            icicle_df,
            path_cols,
            final_path,
            metric,
            METRIC_CONFIG[metric]["label"],
        )

    return selected_sublogs

def chart_selecting(result_set_name, log_view):
    """
    Interactive selector for Icicle and Pie charts across all metrics.
    User selects via checkboxes and clicks 'Go' to render.
    """

    checkboxes = []

    for metric, config in METRIC_CONFIG.items():
        checkboxes.append((
            widgets.Checkbox(value=False, description=f"Icicle – {config['label']}"),
            ("icicle", metric)
        ))

        checkboxes.append((
            widgets.Checkbox(value=False, description=f"Pie – {config['label']}"),
            ("pie", metric)
        ))

    reorderable_rows = []

    for i in range(3):
        cb = widgets.Checkbox(
            value=False,
            description=f"Reorderable Icicle {i + 1}"
        )

        metric_dropdown = widgets.Dropdown(
            options=[(config["label"], metric) for metric, config in METRIC_CONFIG.items()],
            value="avg_case_duration_seconds",
            description="Metric"
        )

        reorderable_rows.append((cb, metric_dropdown))

    checkbox_widgets = [cb for cb, _ in checkboxes]
    reorderable_widgets = [widgets.HBox([cb, dropdown]) for cb, dropdown in reorderable_rows]

    go_button = widgets.Button(
        description="Go",
        button_style="success",
        tooltip="Generate selected visualizations",
        icon="play"
    )

    out = widgets.Output()

    def on_click(b):
        with out:
            clear_output(wait=True)

            selected = [info for cb, info in checkboxes if cb.value]

            for cb, metric_dropdown in reorderable_rows:
                if cb.value:
                    selected.append(("reorderable_icicle", metric_dropdown.value))

            if not selected:
                print("⚠️ Please select at least one visualization.")
                return

            for cb in checkbox_widgets:
                cb.disabled = True
            for cb, dropdown in reorderable_rows:
                cb.disabled = True
                dropdown.disabled = True
            go_button.disabled = True

            labels = [
                f"{kind.title()} – {METRIC_CONFIG[metric]['label']}"
                for kind, metric in selected
            ]
            print(f"⏳ Loading charts: {', '.join(labels)}...\n")

            try:
                for kind, metric in selected:
                    if kind == "icicle":
                        query_exploration_icicle(
                            result_set_name,
                            log_view,
                            metric=metric,
                            details=False
                        )
                    elif kind == "pie":
                        query_breakdown_pie(
                            result_set_name,
                            log_view,
                            metric=metric,
                            details=False
                        )
                    elif kind == "reorderable_icicle":
                        interactive_reorderable_icicle(
                            result_set_name,
                            log_view,
                            metric=metric
                        )
            finally:
                for cb in checkbox_widgets:
                    cb.disabled = False
                for cb, dropdown in reorderable_rows:
                    cb.disabled = False
                    dropdown.disabled = False
                go_button.disabled = False

    go_button.on_click(on_click)

    ui = widgets.VBox([
        widgets.HTML("<h3>Select Visualizations (you can tick multiple)</h3>"),
        widgets.VBox(checkbox_widgets + reorderable_widgets),
        go_button,
        out
    ])

    display(ui)


def interactive_icicle(result_set_name, log_view):
    """Interactive icicle chart with dropdown to switch between metrics."""
    lineage = get_lineage(log_view.query_registry.summary(), result_set_name)

    path_cols = None
    traces = []
    buttons = []

    for metric in METRIC_CONFIG.keys():
        icicle_df, _ = apply_filters(lineage, log_view, metric)
        path_cols = [c for c in icicle_df.columns if c.startswith("Level")]

        icicle_df["hover_text"] = icicle_df.apply(
            lambda row: (
                f"<b>{int(row['num_cases']):,} cases</b><br>"
                f"{METRIC_CONFIG[metric]['label']}: {format_metric_value(metric, row[metric])}"
            ),
            axis=1,
        )

        fig_metric = px.icicle(
            icicle_df,
            path=path_cols,
            values="num_cases",
            color=metric,
            custom_data=["hover_text"],
            color_continuous_scale=METRIC_CONFIG[metric]["color_scheme"],
        )

        trace = fig_metric.data[0]
        trace.visible = metric == "avg_case_duration_seconds"
        traces.append(trace)

        buttons.append(
            dict(
                label=METRIC_CONFIG[metric]["label"],
                method="update",
                args=[
                    {"visible": [t == trace for t in traces]},
                    {"coloraxis": {"colorbar": {"title": METRIC_CONFIG[metric]["label"]}}},
                ],
            )
        )

    fig = go.Figure(data=traces)
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

    fig.update_layout(
        title=f"Icicle Chart for: {result_set_name}",
        margin=dict(t=40, l=0, r=0, b=0),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=1.05,
            y=1.0,
        )],
    )

    fig.show()


def interactive_reorderable_icicle(result_set_name, log_view, metric="avg_case_duration_seconds"):
    """
    Interactive icicle chart where the user can select filters from the
    current registry, reorder only the selected filters, choose a metric,
    and add multiple visuals to a comparison dashboard.
    """
    matrix, base_df = build_registry_filter_matrix(log_view)
    filter_names = list(matrix.columns)

    available_filters_widget = widgets.SelectMultiple(
        options=filter_names,
        value=(),
        description="Available filters",
        rows=len(filter_names),
    )

    order_widget = widgets.Select(
        options=[],
        value=None,
        description="Selected filters",
        rows=len(filter_names),
    )

    metric_widget = widgets.Dropdown(
        options=[(config["label"], metric_key) for metric_key, config in METRIC_CONFIG.items()],
        value=metric,
        description="Metric",
    )

    add_filter_button = widgets.Button(
        description="Add filters",
        button_style="info"
    )

    remove_filter_button = widgets.Button(
        description="Remove filter",
        button_style="warning"
    )

    up_button = widgets.Button(description="Move up")
    down_button = widgets.Button(description="Move down")

    add_button = widgets.Button(
        description="Add to dashboard",
        button_style="success"
    )

    clear_button = widgets.Button(
        description="Clear dashboard",
        button_style="warning"
    )

    message_out = widgets.Output()
    dashboard = widgets.VBox(
        [],
        layout=widgets.Layout(
            width="100%",
            align_items="stretch"
        )
    )
    visual_cards = []

    def current_order():
        return list(order_widget.options)

    def on_add_filters(_):
        current = list(order_widget.options)

        for filter_name in available_filters_widget.value:
            if filter_name not in current:
                current.append(filter_name)

        order_widget.options = current
        order_widget.value = current[0] if current else None

        with message_out:
            clear_output(wait=True)
            if current:
                print("Added selected filters. You can now reorder them in Step 2.")
            else:
                print("Select one or more filters first.")

    def on_remove_filter(_):
        selected = order_widget.value

        if selected is None:
            with message_out:
                clear_output(wait=True)
                print("Select a filter from the selected-filter list first.")
            return

        current = [f for f in order_widget.options if f != selected]
        order_widget.options = current
        order_widget.value = current[0] if current else None

        with message_out:
            clear_output(wait=True)
            print(f"Removed filter: {selected}")

    def create_visual_card():
        order = current_order()
        selected_metric = metric_widget.value

        if not order:
            with message_out:
                clear_output(wait=True)
                print("Please add at least one filter before creating a visual.")
            return

        icicle_df = apply_filters_from_matrix(matrix, base_df, order, selected_metric)
        path_cols = [c for c in icicle_df.columns if c.startswith("Level")]

        icicle_df["hover_text"] = icicle_df.apply(
            lambda row: (
                f"<b>{int(row['num_cases']):,} cases</b><br>"
                f"{METRIC_CONFIG[selected_metric]['label']}: "
                f"{format_metric_value(selected_metric, row[selected_metric])}"
            ),
            axis=1,
        )

        fig = px.icicle(
            icicle_df,
            path=path_cols,
            values="num_cases",
            color=selected_metric,
            custom_data=["hover_text"],
            color_continuous_scale=METRIC_CONFIG[selected_metric]["color_scheme"],
            title=(
                f"{METRIC_CONFIG[selected_metric]['label']}<br>"
                f"Order: {' → '.join(order)}"
            ),
        )

        fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
        fig.update_layout(
            margin=dict(t=60, l=20, r=20, b=20),
            width=1300,          # or 1400 if your notebook is very wide
            height=650,
            coloraxis_colorbar=dict(
                title=METRIC_CONFIG[selected_metric]["label"]
            ),
        )

        chart_out = widgets.Output(
            layout=widgets.Layout(width="1300px")
        )
        with chart_out:
            fig.show()

        remove_button = widgets.Button(
            description="Remove",
            button_style="danger",
            icon="trash"
        )

        card = widgets.VBox(
            [
                widgets.HTML(
                    f"<b>Metric:</b> {METRIC_CONFIG[selected_metric]['label']}<br>"
                    f"<b>Filters:</b> {' → '.join(order)}"
                ),
                remove_button,
                chart_out,
            ],
            layout=widgets.Layout(width="100%")
        )

        def remove_card(_):
            if card in visual_cards:
                visual_cards.remove(card)
                dashboard.children = tuple(visual_cards)

        remove_button.on_click(remove_card)

        visual_cards.append(card)
        dashboard.children = tuple(visual_cards)

        with message_out:
            clear_output(wait=True)
            print(
                f"Added visual to dashboard | "
                f"Metric: {METRIC_CONFIG[selected_metric]['label']} | "
                f"Filters: {' → '.join(order)}"
            )

    def move_selected(delta):
        selected = order_widget.value
        options = list(order_widget.options)

        if selected is None:
            with message_out:
                clear_output(wait=True)
                print("Select a filter to move.")
            return

        idx = options.index(selected)
        new_idx = idx + delta

        if new_idx < 0:
            with message_out:
                clear_output(wait=True)
                print(f"'{selected}' is already at the top.")
            return

        if new_idx >= len(options):
            with message_out:
                clear_output(wait=True)
                print(f"'{selected}' is already at the bottom.")
            return

        options[idx], options[new_idx] = options[new_idx], options[idx]

        order_widget.options = options
        order_widget.value = selected

    def on_up(_):
        move_selected(-1)

    def on_down(_):
        move_selected(1)

    def on_add(_):
        create_visual_card()

    def on_clear(_):
        visual_cards.clear()
        dashboard.children = tuple()
        with message_out:
            clear_output(wait=True)
            print("Cleared all visuals from the dashboard.")

    add_filter_button.on_click(on_add_filters)
    remove_filter_button.on_click(on_remove_filter)
    up_button.on_click(on_up)
    down_button.on_click(on_down)
    add_button.on_click(on_add)
    clear_button.on_click(on_clear)

    ui = widgets.VBox([
        widgets.HTML("<h3>Build comparison dashboard</h3>"),

        widgets.HTML("""
        <h4>Step 1 – Select filters to include</h4>
        <p>
        Select one or more filters from the registry. These filters are not visualized yet;
        they will first be added to the reorderable list.
        </p>
        """),
        available_filters_widget,
        add_filter_button,

        widgets.HTML("""
        <hr>
        <h4>Step 2 – Reorder selected filters</h4>
        <p>
        Only the filters added in Step 1 appear here. Move them up or down to define
        the hierarchy used in the icicle chart. You can also remove a selected filter.
        </p>
        """),
        order_widget,
        widgets.HBox([up_button, down_button, remove_filter_button]),

        widgets.HTML("""
        <hr>
        <h4>Step 3 – Choose metric</h4>
        <p>
        Choose which metric should determine the coloring of the icicle.
        The same filter order can be added multiple times with different metrics.
        </p>
        """),
        metric_widget,

        widgets.HTML("""
        <hr>
        <h4>Step 4 – Add visual to dashboard</h4>
        <p>
        Click <b>Add to dashboard</b> to add the current filter selection, order,
        and metric as a new icicle. Repeat the steps to compare multiple visuals.
        </p>
        """),
        widgets.HBox([add_button, clear_button]),

        message_out,

        widgets.HTML("<h4>Comparison dashboard</h4>"),
        dashboard,
    ])

    display(ui)

def interactive_pie(result_set_name, log_view):
    """Interactive pie chart with dropdown to switch between metrics."""
    traces = []
    buttons = []

    for metric, config in METRIC_CONFIG.items():
        full_log = log_view.query_registry.get_initial_source_log()
        full_log = config["enrich_fn"](full_log)
        value_col, color_scheme, color_title = (
            config["column"],
            config["color_scheme"],
            config["label"],
        )

        parent_log, query_obj, label, step_index, lineage_df = get_sibling_subsets(
            result_set_name, log_view
        )
        filtered, _ = log_view.query_evaluator.evaluate(full_log, query_obj)
        if filtered.empty:
            continue

        case_paths, final_result_path = build_case_paths(
            lineage_df, full_log, filtered, log_view
        )

        filtered = filtered.copy()
        filtered["path_label"] = filtered["case:concept:name"].map(
            lambda cid: " → ".join(case_paths.get(cid, []))
        )

        grouped = (
            filtered.groupby("path_label")[["case:concept:name", value_col]]
            .agg(num_cases=("case:concept:name", "nunique"), avg_metric=(value_col, "mean"))
            .reset_index()
        )
        grouped = grouped.copy()
        grouped["wrapped_path"] = grouped["path_label"].str.replace(" → ", " →<br>")
        grouped = grouped.rename(columns={"avg_metric": metric})

        color_values = get_normalized_colors(grouped[metric], color_scheme)
        grouped = format_slice_labels(grouped, final_result_path)

        pie_trace = go.Pie(
            labels=grouped["slice_label"],
            values=grouped["num_cases"],
            textinfo="label",
            customdata=grouped[["wrapped_path"]],
            hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
            marker=dict(colors=color_values),
            visible=metric == "avg_case_duration_seconds",
        )

        traces.append(pie_trace)

        buttons.append(
            dict(
                label=config["label"],
                method="update",
                args=[
                    {"visible": [t == pie_trace for t in traces]},
                    {"title": f"Pie Chart for: {query_obj.as_string()}<br>({config['label']})"},
                ],
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Pie Chart for: {result_set_name}",
        width=800,
        height=700,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=1.05,
            y=1.0,
        )],
    )

    fig.show()