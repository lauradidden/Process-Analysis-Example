"""Interactive, clickable, reorderable filter-branch icicle dashboard.
All filters must already be registered in the supplied LogView.
The ``filters`` argument accepts registered query names only. """

import keyword

import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import clear_output, display

from utils import METRIC_CONFIG, build_query_maps, format_metric_value
from lineage_core import get_lineage


def filterbranchmap(
    log_view,
    result_set_name=None,
    metric="avg_case_duration_seconds",
    custom_metrics=None,
    filters=None,
    source_log=None,
    case_col="case:concept:name",
    missing_metric="drop",
    allow_filter_selection=True,
    allow_reordering=True,
    allow_subset_saving=True,
):
    """Build a multi-chart filter-branch comparison dashboard.

    ``metric`` may be a standard ``METRIC_CONFIG`` key, the name of an entry
    in ``custom_metrics``, or a custom metric definition supplied directly.

    A custom metric definition has this form::

        {
            "label": "Average resource handoffs",
            "formula": calculate_resource_handoffs,
            "color_scheme": "YlOrRd",
            "aggregation": "mean",
            "color_min": 0,
            "color_max": 100,
        }

    Supplying ``color_min`` and ``color_max`` locks the color normalization
    and legend range. This makes colors comparable across separately created
    charts. Either supply both limits or neither.

    Its formula must return one numeric value per case, as a Series indexed by
    case ID, a mapping, or a DataFrame containing ``case_col`` and
    ``metric_value``. Filters are evaluated independently on the source log;
    reordering changes the displayed hierarchy, not filter semantics.

    The returned dictionary is populated interactively when subsets are saved.
    In IPython, each saved DataFrame is also exposed under the chosen variable
    name.
    """

    if missing_metric not in {"drop", "error", "zero"}:
        raise ValueError("missing_metric must be 'drop', 'error', or 'zero'.")

    custom_metrics = {} if custom_metrics is None else dict(custom_metrics)
    query_map, query_expr_map = build_query_maps(log_view.query_registry)
    if not query_map:
        raise ValueError("No filters were found in the query registry.")

    # Resolve the original event log and, when available, the result lineage.
    lineage = None
    if result_set_name is not None:
        lineage = get_lineage(log_view.query_registry.summary(), result_set_name)

    if source_log is not None:
        base_df = source_log.copy()
    elif lineage is not None and not lineage.empty:
        source_name = lineage.iloc[0].get("source_log")
        cache = getattr(log_view, "result_set_name_cache", {})
        if source_name is not None and source_name in cache:
            base_df = cache[source_name].copy()
        else:
            base_df = log_view.query_registry.get_initial_source_log().copy()
    else:
        base_df = log_view.query_registry.get_initial_source_log().copy()

    if case_col not in base_df.columns:
        raise KeyError(f"The source log has no '{case_col}' column.")
    if base_df.empty:
        raise ValueError("The source event log is empty.")

    source_case_ids = pd.Index(base_df[case_col].dropna().unique(), name=case_col)
    if source_case_ids.empty:
        raise ValueError("The source event log contains no valid case IDs.")

    def resolve_filter(reference):
        if not isinstance(reference, str):
            raise TypeError(
                "Filters must be provided as registered LogView query names."
            )

        if reference not in query_map:
            raise KeyError(
                f"Filter {reference!r} is not registered in the supplied LogView."
            )

        return reference

    def extract_filters_from_lineage(lineage_df):
        if lineage_df is None or lineage_df.empty:
            return []

        names = []
        for _, row in lineage_df.iterrows():
            row_values = []
            for value in row.tolist():
                # pd.isna(array) returns an array, so only test scalar values.
                if pd.api.types.is_scalar(value) and pd.isna(value):
                    continue
                text = str(value).strip()
                if text:
                    row_values.append(text)

            match = None
            for query_name in query_map:
                expression = query_expr_map.get(query_name)
                if query_name in row_values or (expression and expression in row_values):
                    match = query_name
                    break

            if match is None:
                combined = " ".join(row_values)
                for query_name in query_map:
                    expression = query_expr_map.get(query_name)
                    if expression and expression in combined:
                        match = query_name
                        break

            if match is not None and match not in names:
                names.append(match)
        return names

    if filters is not None:
        initial_filter_names = [resolve_filter(item) for item in filters]
    else:
        initial_filter_names = extract_filters_from_lineage(lineage)
    initial_filter_names = list(dict.fromkeys(initial_filter_names))
    available_filter_names = list(query_map)

    # Build a catalog containing all standard and supplied custom metrics.
    metric_catalog = {
        name: {
            "kind": "standard",
            "name": name,
            "label": config["label"],
            "definition": config,
        }
        for name, config in METRIC_CONFIG.items()
    }

    for name, definition in custom_metrics.items():
        if not isinstance(definition, dict):
            raise TypeError(f"Custom metric '{name}' must be a dictionary.")
        if "label" not in definition or "formula" not in definition:
            raise KeyError(f"Custom metric '{name}' needs 'label' and 'formula'.")
        if not callable(definition["formula"]):
            raise TypeError(f"Custom metric '{name}' formula must be callable.")
        metric_catalog[name] = {
            "kind": "custom",
            "name": name,
            "label": definition["label"],
            "definition": definition,
        }

    if isinstance(metric, dict):
        if "label" not in metric or "formula" not in metric:
            raise KeyError("A direct custom metric needs 'label' and 'formula'.")
        initial_metric_name = metric.get("name", "custom_metric")
        original_name = initial_metric_name
        suffix = 2
        while initial_metric_name in metric_catalog:
            initial_metric_name = f"{original_name}_{suffix}"
            suffix += 1
        metric_catalog[initial_metric_name] = {
            "kind": "custom",
            "name": initial_metric_name,
            "label": metric["label"],
            "definition": metric,
        }
    elif isinstance(metric, str):
        if metric not in metric_catalog:
            raise KeyError(
                f"Unknown metric '{metric}'. Available: {list(metric_catalog)}"
            )
        initial_metric_name = metric
    else:
        raise TypeError("metric must be a metric name or custom metric dictionary.")

    def normalize_metric_result(result):
        if isinstance(result, pd.Series):
            result = result.rename("metric_value").rename_axis(case_col).reset_index()
        elif isinstance(result, dict):
            result = pd.DataFrame(
                {case_col: list(result), "metric_value": list(result.values())}
            )
        elif isinstance(result, pd.DataFrame):
            result = result.copy()
            if case_col not in result:
                raise ValueError(f"Metric DataFrame must contain '{case_col}'.")
            if "metric_value" not in result:
                candidates = [column for column in result if column != case_col]
                if len(candidates) != 1:
                    raise ValueError(
                        "Metric DataFrame needs 'metric_value' or exactly one value column."
                    )
                result = result.rename(columns={candidates[0]: "metric_value"})
            result = result[[case_col, "metric_value"]]
        else:
            raise TypeError("Metric formula must return a Series, DataFrame, or mapping.")

        result = result.loc[result[case_col].notna(), [case_col, "metric_value"]].copy()
        duplicated = result[case_col].duplicated(keep=False)
        if duplicated.any():
            examples = result.loc[duplicated, case_col].drop_duplicates().head().tolist()
            raise ValueError(
                "A metric must return at most one value per case. "
                f"Duplicate examples: {examples}"
            )
        result["metric_value"] = pd.to_numeric(result["metric_value"], errors="coerce")
        return result

    metric_cache = {}

    def get_case_metric(metric_name):
        if metric_name in metric_cache:
            metric_df, info = metric_cache[metric_name]
            return metric_df.copy(), dict(info)

        entry = metric_catalog[metric_name]
        definition = entry["definition"]

        if entry["kind"] == "standard":
            enriched = definition["enrich_fn"](base_df.copy())
            value_col = definition["column"]
            if value_col not in enriched:
                raise KeyError(
                    f"Metric '{metric_name}' did not produce column '{value_col}'."
                )

            unique_counts = enriched.groupby(case_col)[value_col].nunique(dropna=True)
            inconsistent = unique_counts[unique_counts > 1]
            if not inconsistent.empty:
                examples = inconsistent.index.astype(str).tolist()[:5]
                raise ValueError(
                    f"Standard metric '{metric_name}' has multiple values per case. "
                    f"Examples: {examples}"
                )

            metric_df = normalize_metric_result(
                enriched[[case_col, value_col]]
                .drop_duplicates(subset=[case_col])
                .rename(columns={value_col: "metric_value"})
            )
            info = {
                "name": metric_name,
                "label": definition["label"],
                "color_scheme": definition["color_scheme"],
                "aggregation": definition.get("aggregation", "mean"),
                "color_min": definition.get("color_min"),
                "color_max": definition.get("color_max"),
                "standard": True,
            }
        else:
            metric_df = normalize_metric_result(definition["formula"](base_df.copy()))
            info = {
                "name": metric_name,
                "label": definition["label"],
                "color_scheme": definition.get("color_scheme", "Viridis"),
                "aggregation": definition.get("aggregation", "mean"),
                "color_min": definition.get("color_min"),
                "color_max": definition.get("color_max"),
                "standard": False,
            }

        if info["aggregation"] not in {"mean", "median", "sum", "min", "max"}:
            raise ValueError(f"Unsupported aggregation: {info['aggregation']}")

        color_min = info["color_min"]
        color_max = info["color_max"]
        if (color_min is None) != (color_max is None):
            raise ValueError(
                f"Metric '{metric_name}' must define both 'color_min' and "
                "'color_max', or neither."
            )
        if color_min is not None and color_min >= color_max:
            raise ValueError(
                f"Metric '{metric_name}' requires color_min < color_max."
            )

        unknown_cases = set(metric_df[case_col]) - set(source_case_ids)
        if unknown_cases:
            raise ValueError(
                "Metric contains cases absent from the source log. "
                f"Examples: {list(unknown_cases)[:5]}"
            )

        metric_cache[metric_name] = (metric_df.copy(), dict(info))
        return metric_df, info

    # Evaluate filters once. Reordering only rebuilds the hierarchy.
    filter_case_sets = {}
    for filter_name in available_filter_names:
        passed_df, _ = log_view.query_evaluator.evaluate(
            base_df.copy(), query_map[filter_name]
        )
        if case_col not in passed_df:
            raise KeyError(f"Filter '{filter_name}' returned no '{case_col}' column.")
        filter_case_sets[filter_name] = set(passed_df[case_col].dropna().unique())

    case_matrix = pd.DataFrame({case_col: source_case_ids})
    for filter_name in available_filter_names:
        case_matrix[filter_name] = case_matrix[case_col].isin(
            filter_case_sets[filter_name]
        )

    selected_sublogs = {}
    pending_selection = {
        "data": None,
        "case_ids": None,
        "path": None,
        "node_id": None,
        "filter_order": None,
        "metric": None,
    }
    visual_cards = []

    available_widget = widgets.SelectMultiple(
        options=available_filter_names,
        value=(),
        description="Available",
        rows=max(5, min(len(available_filter_names), 15)),
        layout=widgets.Layout(width="700px"),
    )
    order_widget = widgets.Select(
        options=initial_filter_names,
        value=initial_filter_names[0] if initial_filter_names else None,
        description="Selected",
        rows=max(5, min(max(len(initial_filter_names), 1), 15)),
        layout=widgets.Layout(width="700px"),
    )
    metric_widget = widgets.Dropdown(
        options=[
            (entry["label"], name) for name, entry in metric_catalog.items()
        ],
        value=initial_metric_name,
        description="Metric",
        layout=widgets.Layout(width="700px"),
    )

    add_filter_button = widgets.Button(description="Add filters", button_style="info")
    remove_filter_button = widgets.Button(
        description="Remove filter", button_style="warning"
    )
    up_button = widgets.Button(description="Move up")
    down_button = widgets.Button(description="Move down")
    add_chart_button = widgets.Button(
        description="Add chart", button_style="success", icon="plus"
    )
    clear_charts_button = widgets.Button(
        description="Clear charts", button_style="warning", icon="trash"
    )

    name_input = widgets.Text(
        value="",
        placeholder="Example: high_handoff_cases",
        description="Name",
        layout=widgets.Layout(width="520px"),
    )
    save_button = widgets.Button(
        description="Save subset", button_style="success", disabled=True
    )

    message_out = widgets.Output()
    preview_out = widgets.Output()
    dashboard = widgets.VBox(
        [], layout=widgets.Layout(width="100%", align_items="stretch")
    )

    def current_order():
        return list(order_widget.options)

    def aggregate_metric(values, aggregation):
        values = values.dropna()
        if values.empty:
            return float("nan")
        return getattr(values, aggregation)()

    def build_node_table(filter_order, chart_cases, metric_info):
        node_case_ids = {}
        node_paths = {}
        records = []
        counter = 0

        def add_node(parent_id, label, path, subset, depth):
            nonlocal counter
            node_id = f"node_{counter}"
            counter += 1
            ids = set(subset[case_col])
            node_case_ids[node_id] = ids
            node_paths[node_id] = list(path)
            values = subset["metric_value"]
            records.append(
                {
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "label": label,
                    "depth": depth,
                    "num_cases": len(ids),
                    "num_metric_cases": int(values.notna().sum()),
                    "metric_value": aggregate_metric(
                        values, metric_info["aggregation"]
                    ),
                    "path": " → ".join(path),
                }
            )
            return node_id

        root_label = "⭐ Initial Source"
        root_id = add_node("", root_label, [root_label], chart_cases, 0)

        def add_children(parent_id, subset, position, path, on_main_path):
            if position >= len(filter_order):
                return
            filter_name = filter_order[position]
            expression = query_expr_map.get(filter_name, filter_name)
            for passed, symbol in ((True, "✔"), (False, "✘")):
                branch = subset.loc[subset[filter_name] == passed].copy()
                if branch.empty:
                    continue
                label = f"{expression} {symbol}"
                branch_on_main_path = on_main_path and passed
                if branch_on_main_path:
                    label = f"⭐ {label}"
                branch_path = path + [label]
                child_id = add_node(
                    parent_id, label, branch_path, branch, position + 1
                )
                add_children(
                    child_id,
                    branch,
                    position + 1,
                    branch_path,
                    branch_on_main_path,
                )

        add_children(root_id, chart_cases, 0, [root_label], True)
        return pd.DataFrame(records), node_case_ids, node_paths

    def format_value(metric_info, value):
        if pd.isna(value):
            return "No metric values"
        if metric_info["standard"]:
            try:
                return format_metric_value(metric_info["name"], value)
            except Exception:
                pass
        return f"{value:,.2f}"

    def add_chart():
        filter_order = current_order()
        if not filter_order:
            with message_out:
                clear_output(wait=True)
                print("Add at least one filter before creating a chart.")
            return

        selected_metric = metric_widget.value
        metric_df, metric_info = get_case_metric(selected_metric)
        chart_cases = case_matrix.merge(
            metric_df, on=case_col, how="left", validate="one_to_one"
        )
        missing_count = int(chart_cases["metric_value"].isna().sum())
        if missing_count and missing_metric == "error":
            raise ValueError(
                f"'{metric_info['label']}' is missing for {missing_count:,} cases."
            )
        if missing_metric == "zero":
            chart_cases["metric_value"] = chart_cases["metric_value"].fillna(0.0)

        node_df, chart_node_cases, chart_node_paths = build_node_table(
            filter_order, chart_cases, metric_info
        )
        node_df["hover_text"] = node_df.apply(
            lambda row: (
                f"<b>{row['path']}</b><br>"
                f"{int(row['num_cases']):,} cases<br>"
                f"Cases with metric: {int(row['num_metric_cases']):,}<br>"
                f"{metric_info['label']}: "
                f"{format_value(metric_info, row['metric_value'])}"
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
                customdata=node_df[["hover_text", "node_id", "path"]].to_numpy(),
                hovertemplate=(
                    "%{customdata[0]}"
                    "<br><i>Click to preview this subset</i><extra></extra>"
                ),
                marker=dict(
                    colors=node_df["metric_value"],
                    colorscale=metric_info["color_scheme"],
                    cmin=metric_info["color_min"],
                    cmax=metric_info["color_max"],
                    cauto=not (
                        metric_info["color_min"] is not None
                        and metric_info["color_max"] is not None
                    ),
                    colorbar=dict(title=metric_info["label"]),
                ),
                root=dict(color="lightgrey"),
                tiling=dict(orientation="h", pad=2),
            )
        )
        fig.update_layout(
            title=(
                f"{metric_info['label']}<br>"
                f"Order: {' → '.join(filter_order)}"
            ),
            margin=dict(t=90, l=20, r=20, b=20),
            width=1200,
            height=700,
        )

        def on_chart_click(trace, points, selector):
            if not points.point_inds:
                return
            node_id = str(trace.ids[points.point_inds[0]])
            selected_ids = chart_node_cases.get(node_id, set())
            subset = base_df.loc[base_df[case_col].isin(selected_ids)].copy()
            path = " → ".join(chart_node_paths.get(node_id, []))
            pending_selection.update(
                {
                    "data": subset,
                    "case_ids": set(selected_ids),
                    "path": path,
                    "node_id": node_id,
                    "filter_order": list(filter_order),
                    "metric": dict(metric_info),
                }
            )
            save_button.disabled = not allow_subset_saving
            with preview_out:
                clear_output(wait=True)
                print(f"Selected node: {path}")
                print(f"Chart metric: {metric_info['label']}")
                print(f"{len(selected_ids):,} cases and {len(subset):,} events.")
                display(subset.head(10))

        fig.data[0].on_click(on_chart_click)

        remove_button = widgets.Button(
            description="Remove chart", button_style="danger", icon="trash"
        )
        card = widgets.VBox(
            [
                widgets.HTML(
                    f"<b>Metric:</b> {metric_info['label']}<br>"
                    f"<b>Aggregation:</b> {metric_info['aggregation']}<br>"
                    f"<b>Filters:</b> {' → '.join(filter_order)}"
                ),
                remove_button,
                fig,
            ],
            layout=widgets.Layout(
                width="100%",
                border="1px solid #dddddd",
                padding="10px",
                margin="0 0 15px 0",
            ),
        )

        def remove_chart(_):
            if card in visual_cards:
                visual_cards.remove(card)
                dashboard.children = tuple(visual_cards)

        remove_button.on_click(remove_chart)
        visual_cards.append(card)
        dashboard.children = tuple(visual_cards)

        with message_out:
            clear_output(wait=True)
            print(
                f"Added chart {len(visual_cards)} | "
                f"Metric: {metric_info['label']} | "
                f"Filters: {' → '.join(filter_order)}"
            )
            if missing_count:
                print(f"{missing_count:,} cases have no value for this metric.")

    def on_add_filters(_):
        order = current_order()
        for name in available_widget.value:
            if name not in order:
                order.append(name)
        order_widget.options = order
        order_widget.value = order[0] if order else None

    def on_remove_filter(_):
        selected = order_widget.value
        if selected is None:
            return
        order = [name for name in current_order() if name != selected]
        order_widget.options = order
        order_widget.value = order[0] if order else None

    def move_filter(delta):
        selected = order_widget.value
        order = current_order()
        if selected is None:
            return
        old_index = order.index(selected)
        new_index = old_index + delta
        if not 0 <= new_index < len(order):
            return
        order[old_index], order[new_index] = order[new_index], order[old_index]
        order_widget.options = order
        order_widget.value = selected

    def on_save_subset(_):
        subset = pending_selection["data"]
        if subset is None:
            with preview_out:
                clear_output(wait=True)
                print("Click a chart node before saving.")
            return

        name = name_input.value.strip()
        if not name:
            with preview_out:
                clear_output(wait=True)
                print("Enter a subset name.")
            return
        if not name.isidentifier() or keyword.iskeyword(name):
            with preview_out:
                clear_output(wait=True)
                print(f"'{name}' is not a valid Python variable name.")
            return

        ip = get_ipython()
        if ip is not None and name in ip.user_ns:
            with preview_out:
                clear_output(wait=True)
                print(f"'{name}' already exists. Choose another name.")
            return

        saved = subset.copy()
        selected_sublogs[name] = {
            "data": saved,
            "case_ids": sorted(pending_selection["case_ids"], key=str),
            "path": pending_selection["path"],
            "node_id": pending_selection["node_id"],
            "filter_order": list(pending_selection["filter_order"]),
            "metric": dict(pending_selection["metric"]),
            "name": name,
        }
        if ip is not None:
            ip.user_ns[name] = saved

        with preview_out:
            clear_output(wait=True)
            print(f"Subset saved as: {name}")
            print(f"Path: {pending_selection['path']}")
            print(f"{saved[case_col].nunique():,} cases and {len(saved):,} events.")
            display(saved.head(10))

    def on_clear_charts(_):
        visual_cards.clear()
        dashboard.children = tuple()
        with message_out:
            clear_output(wait=True)
            print("Cleared all comparison charts.")

    add_filter_button.on_click(on_add_filters)
    remove_filter_button.on_click(on_remove_filter)
    up_button.on_click(lambda _: move_filter(-1))
    down_button.on_click(lambda _: move_filter(1))
    add_chart_button.on_click(lambda _: add_chart())
    clear_charts_button.on_click(on_clear_charts)
    save_button.on_click(on_save_subset)

    controls = [widgets.HTML("<h3>Filter Branch Map</h3>")]
    if allow_filter_selection:
        controls.extend(
            [
                widgets.HTML("<h4>1. Select filters</h4>"),
                available_widget,
                add_filter_button,
            ]
        )

    controls.extend([widgets.HTML("<h4>2. Reorder filters</h4>"), order_widget])
    order_buttons = []
    if allow_reordering:
        order_buttons.extend([up_button, down_button])
    if allow_filter_selection:
        order_buttons.append(remove_filter_button)
    if order_buttons:
        controls.append(widgets.HBox(order_buttons))

    controls.extend(
        [
            widgets.HTML("<h4>3. Choose the metric for the next chart</h4>"),
            metric_widget,
            widgets.HBox([add_chart_button, clear_charts_button]),
            message_out,
        ]
    )

    # Save controls deliberately appear above the chart dashboard.
    if allow_subset_saving:
        controls.extend(
            [
                widgets.HTML("<hr><h4>Clicked subset: preview and save</h4>"),
                widgets.HTML(
                    "Click a node in any chart, then name and save its original event rows."
                ),
                widgets.HBox([name_input, save_button]),
                preview_out,
            ]
        )
    else:
        controls.extend([widgets.HTML("<h4>Clicked subset preview</h4>"), preview_out])

    controls.extend(
        [
            widgets.HTML("<hr><h4>Comparison dashboard</h4>"),
            dashboard,
        ]
    )

    display(
        widgets.VBox(
            controls,
            layout=widgets.Layout(width="100%", align_items="stretch"),
        )
    )

    if initial_filter_names:
        add_chart()
    else:
        with message_out:
            print("Select filters and click 'Add chart'.")

    return selected_sublogs

