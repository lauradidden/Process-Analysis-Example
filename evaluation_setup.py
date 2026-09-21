"""Helper functions for the ProMiSE P1 FilterBranchMap evaluation notebook.

The reusable predicates live in ``queries.py``. This module contains the data
preparation, LogView registration, metric, and analysis helpers that those
predicates need.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from logview.utils import LogViewBuilder
from queries import (
    query_create_eventually_send,
    query_credit_collection_violation,
    query_dismissal_is_g,
    query_dismissal_is_only_nil,
    query_dismissal_is_prefecture,
    query_first_four_variants,
    query_first_six_reference_variants,
    query_first_two_variants,
    query_judge_dismissed,
    query_more_than_87_days,
    query_nonconforming,
    query_notification_to_judge_appeal,
    query_notification_to_prefecture_appeal,
    query_prefecture_dismissed,
    query_with_credit_collection,
    query_with_payment,
    query_without_credit_collection,
    query_without_payment,
)


PM_CASE_ID = "case:concept:name"
PM_ACTIVITY = "concept:name"
PM_TIMESTAMP = "time:timestamp"


def build_evaluation_log(log: pd.DataFrame):
    """Copy and sort the event log, then build its shared LogView."""
    source = (
        log.copy()
        .sort_values([PM_CASE_ID, PM_TIMESTAMP], kind="stable")
        .reset_index(drop=True)
    )
    source.name = "FullLog"
    log_view = LogViewBuilder.build_log_view(source)
    return source, log_view


def _name_pair(result, complement, result_name, complement_name):
    result.name = result_name
    complement.name = complement_name
    return result, complement


def add_variant_rank(source: pd.DataFrame) -> pd.DataFrame:
    """Add the global frequency-based variant rank to every event in place."""
    case_variants = source.groupby(PM_CASE_ID, sort=False)[PM_ACTIVITY].apply(tuple)
    counts = (
        case_variants.value_counts()
        .rename_axis("variant")
        .reset_index(name="case_count")
    )
    counts["variant_rank"] = counts.index + 1
    lookup = dict(zip(counts["variant"], counts["variant_rank"]))
    source["variant_rank"] = source[PM_CASE_ID].map(case_variants.map(lookup))
    source.name = "FullLog"
    return counts


def register_variant_stack(log_view, source: pd.DataFrame) -> dict[str, Any]:
    """Register all variants -> first four -> first two."""
    variant_counts = add_variant_rank(source)
    first_four, after_four = log_view.evaluate_query(
        "rs_FirstFourVariants", source, query_first_four_variants
    )
    _name_pair(
        first_four,
        after_four,
        "rs_FirstFourVariants",
        "complement_VariantsAfterFour",
    )
    first_two, three_and_four = log_view.evaluate_query(
        "rs_FirstTwoVariants", first_four, query_first_two_variants
    )
    _name_pair(
        first_two,
        three_and_four,
        "rs_FirstTwoVariants",
        "complement_VariantsThreeAndFour",
    )
    return {
        "variant_counts": variant_counts,
        "all_variants": source,
        "first_four": first_four,
        "after_four": after_four,
        "first_two": first_two,
        "three_and_four": three_and_four,
        "queries": [query_first_four_variants, query_first_two_variants],
        "result_set_name": "rs_FirstTwoVariants",
    }


def register_without_payment(log_view, first_four: pd.DataFrame) -> dict[str, Any]:
    """Register first four variants -> cases without Payment."""
    without_payment, with_payment = log_view.evaluate_query(
        "rs_WithoutPayment", first_four, query_without_payment
    )
    _name_pair(
        without_payment,
        with_payment,
        "rs_WithoutPayment",
        "complement_WithPayment",
    )
    return {
        "without_payment": without_payment,
        "with_payment": with_payment,
        "queries": [query_first_four_variants, query_without_payment],
        "result_set_name": "rs_WithoutPayment",
    }


def add_reference_variant_rank(source: pd.DataFrame) -> pd.DataFrame:
    """Rank variants among paid cases not sent for credit collection."""
    all_ids = set(source[PM_CASE_ID].unique())
    credit_ids = set(
        source.loc[
            source[PM_ACTIVITY] == "Send for Credit Collection", PM_CASE_ID
        ]
    )
    payment_ids = set(source.loc[source[PM_ACTIVITY] == "Payment", PM_CASE_ID])
    candidate_ids = (all_ids - credit_ids) & payment_ids
    candidates = source.loc[source[PM_CASE_ID].isin(candidate_ids)].copy()
    case_variants = (
        candidates.sort_values([PM_CASE_ID, PM_TIMESTAMP], kind="stable")
        .groupby(PM_CASE_ID)[PM_ACTIVITY]
        .apply(tuple)
    )
    counts = (
        case_variants.value_counts()
        .rename_axis("reference_variant")
        .reset_index(name="case_count")
    )
    counts["reference_variant_rank"] = counts.index + 1
    lookup = dict(zip(counts["reference_variant"], counts["reference_variant_rank"]))
    source["reference_variant_rank"] = source[PM_CASE_ID].map(
        case_variants.map(lookup)
    )
    source.name = "FullLog"
    return counts


def register_reference_model_stack(log_view, source: pd.DataFrame) -> dict[str, Any]:
    """Register no credit collection -> Payment -> first six variants."""
    counts = add_reference_variant_rank(source)
    without_credit, with_credit = log_view.evaluate_query(
        "rs_WithoutCreditCollection", source, query_without_credit_collection
    )
    _name_pair(
        without_credit,
        with_credit,
        "rs_WithoutCreditCollection",
        "complement_WithCreditCollection",
    )
    candidates, candidates_without_payment = log_view.evaluate_query(
        "rs_ReferenceCandidates", without_credit, query_with_payment
    )
    _name_pair(
        candidates,
        candidates_without_payment,
        "rs_ReferenceCandidates",
        "complement_ReferenceWithoutPayment",
    )
    first_six, other_variants = log_view.evaluate_query(
        "rs_FirstSixReferenceVariants",
        candidates,
        query_first_six_reference_variants,
    )
    _name_pair(
        first_six,
        other_variants,
        "rs_FirstSixReferenceVariants",
        "complement_OtherReferenceVariants",
    )
    return {
        "reference_variant_counts": counts,
        "without_credit_collection": without_credit,
        "with_credit_collection": with_credit,
        "reference_candidates": candidates,
        "reference_candidates_without_payment": candidates_without_payment,
        "first_six": first_six,
        "other_variants": other_variants,
        "queries": [
            query_without_credit_collection,
            query_with_payment,
            query_first_six_reference_variants,
        ],
        "result_set_name": "rs_FirstSixReferenceVariants",
    }


def first_later_activity_gap_days(
    event_log: pd.DataFrame,
    first_activity: str,
    second_activity: str,
) -> pd.DataFrame:
    """Return the first-to-first-later activity gap in days for each case."""
    records = []
    for case_id, case in event_log.groupby(PM_CASE_ID):
        case = case.sort_values(PM_TIMESTAMP, kind="stable")
        first_times = case.loc[case[PM_ACTIVITY] == first_activity, PM_TIMESTAMP]
        if first_times.empty:
            continue
        first_time = first_times.iloc[0]
        later = case.loc[
            (case[PM_ACTIVITY] == second_activity)
            & (case[PM_TIMESTAMP] > first_time),
            PM_TIMESTAMP,
        ]
        if later.empty:
            continue
        second_time = later.iloc[0]
        records.append(
            {
                PM_CASE_ID: case_id,
                "first_activity": first_activity,
                "second_activity": second_activity,
                "first_timestamp": first_time,
                "second_timestamp": second_time,
                "days_between": (second_time - first_time).total_seconds() / 86400,
            }
        )
    return pd.DataFrame(records)


def add_delayed_notification_attributes(
    source: pd.DataFrame, conformance_df: pd.DataFrame
) -> None:
    """Add conformance status and Create Fine -> Send Fine duration in place."""
    gaps = first_later_activity_gap_days(source, "Create Fine", "Send Fine")
    gap_by_case = gaps.drop_duplicates(PM_CASE_ID).set_index(PM_CASE_ID)[
        "days_between"
    ]
    source["create_to_send_days"] = source[PM_CASE_ID].map(gap_by_case)

    conformance = (
        conformance_df[["case_id", "conforming"]]
        .drop_duplicates("case_id")
        .set_index("case_id")["conforming"]
        .map({True: "conforming", False: "non-conforming"})
    )
    source["case_is_conforming"] = source[PM_CASE_ID].map(conformance)
    source.name = "FullLog"


def register_delayed_notification_stack(
    log_view, source: pd.DataFrame, conformance_df: pd.DataFrame
) -> dict[str, Any]:
    """Register the four-stage delayed-notification investigation."""
    add_delayed_notification_attributes(source, conformance_df)
    nonconforming, conforming = log_view.evaluate_query(
        "rs_NonConforming", source, query_nonconforming
    )
    _name_pair(nonconforming, conforming, "rs_NonConforming", "complement_Conforming")

    credit, no_credit = log_view.evaluate_query(
        "rs_CreditCollectionViolations",
        nonconforming,
        query_credit_collection_violation,
    )
    _name_pair(
        credit,
        no_credit,
        "rs_CreditCollectionViolations",
        "complement_NonConformingWithoutCreditCollection",
    )

    create_send, no_relation = log_view.evaluate_query(
        "rs_CreateEventuallySend", credit, query_create_eventually_send
    )
    _name_pair(
        create_send,
        no_relation,
        "rs_CreateEventuallySend",
        "complement_WithoutCreateSendRelation",
    )

    delayed, within_87 = log_view.evaluate_query(
        "rs_DelayedNotification", create_send, query_more_than_87_days
    )
    _name_pair(
        delayed,
        within_87,
        "rs_DelayedNotification",
        "complement_CreateToSendWithin87Days",
    )
    return {
        "nonconforming": nonconforming,
        "conforming": conforming,
        "credit_collection_violations": credit,
        "without_credit_collection": no_credit,
        "create_eventually_send": create_send,
        "without_create_send_relation": no_relation,
        "delayed_notification": delayed,
        "within_87_days": within_87,
        "queries": [
            query_nonconforming,
            query_credit_collection_violation,
            query_create_eventually_send,
            query_more_than_87_days,
        ],
        "result_set_name": "rs_DelayedNotification",
    }


def register_prefecture_stack(log_view, source: pd.DataFrame) -> dict[str, Any]:
    """Register notification -> prefecture appeal -> prefecture dismissal."""
    appeals, no_appeal = log_view.evaluate_query(
        "rs_PrefectureAppeals", source, query_notification_to_prefecture_appeal
    )
    _name_pair(
        appeals, no_appeal, "rs_PrefectureAppeals", "complement_WithoutPrefectureAppeal"
    )
    dismissed, not_dismissed = log_view.evaluate_query(
        "rs_PrefectureDismissed", appeals, query_prefecture_dismissed
    )
    _name_pair(
        dismissed,
        not_dismissed,
        "rs_PrefectureDismissed",
        "complement_PrefectureAppealsNotDismissed",
    )
    return {
        "appeals": appeals,
        "without_appeal": no_appeal,
        "dismissed": dismissed,
        "not_dismissed": not_dismissed,
        "queries": [query_notification_to_prefecture_appeal, query_prefecture_dismissed],
        "result_set_name": "rs_PrefectureDismissed",
    }


def register_judge_stack(log_view, source: pd.DataFrame) -> dict[str, Any]:
    """Register notification -> judge appeal -> judge dismissal."""
    appeals, no_appeal = log_view.evaluate_query(
        "rs_JudgeAppeals", source, query_notification_to_judge_appeal
    )
    _name_pair(appeals, no_appeal, "rs_JudgeAppeals", "complement_WithoutJudgeAppeal")
    dismissed, not_dismissed = log_view.evaluate_query(
        "rs_JudgeDismissed", appeals, query_judge_dismissed
    )
    _name_pair(
        dismissed,
        not_dismissed,
        "rs_JudgeDismissed",
        "complement_JudgeAppealsNotDismissed",
    )
    return {
        "appeals": appeals,
        "without_appeal": no_appeal,
        "dismissed": dismissed,
        "not_dismissed": not_dismissed,
        "queries": [query_notification_to_judge_appeal, query_judge_dismissed],
        "result_set_name": "rs_JudgeDismissed",
    }


def classify_nil_dismissal(values: pd.Series) -> str:
    """Classify a case as only_NIL when all observed values are NIL."""
    observed = values.dropna().astype(str).str.strip()
    return "only_NIL" if not observed.empty and observed.eq("NIL").all() else "has_other_value"


def add_nil_dismissal_status(source: pd.DataFrame) -> pd.DataFrame:
    """Add the strict case-level NIL classification in place and summarize it."""
    status = source.groupby(PM_CASE_ID)["dismissal"].agg(classify_nil_dismissal)
    source["nil_dismissal_status"] = source[PM_CASE_ID].map(status)
    source.name = "FullLog"
    return (
        source[[PM_CASE_ID, "nil_dismissal_status"]]
        .drop_duplicates(PM_CASE_ID)["nil_dismissal_status"]
        .value_counts(dropna=False)
        .rename_axis("nil_dismissal_status")
        .reset_index(name="cases")
    )


def evaluate_dismissal_comparison(log_view, source: pd.DataFrame) -> dict[str, Any]:
    """Evaluate credit-collection dismissal categories without registry writes."""
    classification_summary = add_nil_dismissal_status(source)
    with_credit, without_credit = log_view.query_evaluator.evaluate(
        source.copy(), query_with_credit_collection
    )
    with_credit, without_credit = with_credit.copy(), without_credit.copy()
    _name_pair(
        with_credit,
        without_credit,
        "rs_WithCreditCollection",
        "complement_WithoutCreditCollection",
    )

    only_nil, not_only_nil = log_view.query_evaluator.evaluate(
        with_credit.copy(), query_dismissal_is_only_nil
    )
    contains_g, not_g = log_view.query_evaluator.evaluate(
        with_credit.copy(), query_dismissal_is_g
    )
    contains_hash, not_hash = log_view.query_evaluator.evaluate(
        with_credit.copy(), query_dismissal_is_prefecture
    )
    pairs = [
        (only_nil.copy(), not_only_nil.copy(), "rs_CreditCollectionOnlyNIL", "complement_CreditCollectionNotOnlyNIL"),
        (contains_g.copy(), not_g.copy(), "rs_CreditCollectionDismissalG", "complement_CreditCollectionNotG"),
        (contains_hash.copy(), not_hash.copy(), "rs_CreditCollectionDismissalPrefecture", "complement_CreditCollectionNotPrefecture"),
    ]
    for result, complement, result_name, complement_name in pairs:
        _name_pair(result, complement, result_name, complement_name)
    only_nil, not_only_nil = pairs[0][0], pairs[0][1]
    contains_g, not_g = pairs[1][0], pairs[1][1]
    contains_hash, not_hash = pairs[2][0], pairs[2][1]

    summary = pd.DataFrame(
        [
            {"category": "All cases", "cases": source[PM_CASE_ID].nunique()},
            {"category": "Contains Send for Credit Collection", "cases": with_credit[PM_CASE_ID].nunique()},
            {"category": "Credit collection and only NIL", "cases": only_nil[PM_CASE_ID].nunique()},
            {"category": 'Credit collection and contains dismissal = "G"', "cases": contains_g[PM_CASE_ID].nunique()},
            {"category": 'Credit collection and contains dismissal = "#"', "cases": contains_hash[PM_CASE_ID].nunique()},
        ]
    )
    return {
        "classification_summary": classification_summary,
        "summary": summary,
        "with_credit_collection": with_credit,
        "without_credit_collection": without_credit,
        "only_nil": only_nil,
        "not_only_nil": not_only_nil,
        "contains_g": contains_g,
        "not_g": not_g,
        "contains_prefecture": contains_hash,
        "not_prefecture": not_hash,
        "queries": [
            query_with_credit_collection,
            query_dismissal_is_only_nil,
            query_dismissal_is_g,
            query_dismissal_is_prefecture,
        ],
        "result_set_name": None,
    }

def _payment_case_values(log: pd.DataFrame) -> pd.DataFrame:
    required = {
        PM_CASE_ID,
        PM_TIMESTAMP,
        "amount",
        "expense",
        "totalPaymentAmount",
    }
    missing = required - set(log.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    working = log[list(required)].copy()
    working[PM_TIMESTAMP] = pd.to_datetime(
        working[PM_TIMESTAMP], errors="coerce", utc=True
    )

    for column in ["amount", "expense", "totalPaymentAmount"]:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working["_original_order"] = range(len(working))
    working = working.sort_values(
        [PM_CASE_ID, PM_TIMESTAMP, "_original_order"],
        kind="stable",
    )

    values = working.groupby(PM_CASE_ID).agg(
        updated_fine_amount=("amount", "max"),
        expense=("expense", "max"),
        final_total_payment=("totalPaymentAmount", "last"),
    )

    values[["updated_fine_amount", "expense", "final_total_payment"]] = (
        values[["updated_fine_amount", "expense", "final_total_payment"]]
        .fillna(0.0)
    )

    return values

def calculate_outstanding_payment(log: pd.DataFrame) -> pd.Series:
    values = _payment_case_values(log)

    result = (
        values["updated_fine_amount"]
        + values["expense"]
        - values["final_total_payment"]
    ).clip(lower=0)

    return result.rename("metric_value")


def calculate_paid_percentage(log: pd.DataFrame) -> pd.Series:
    """Return 100 for fully paid cases and 0 otherwise."""
    values = _payment_case_values(log)

    balance = (
        values["updated_fine_amount"]
        + values["expense"]
        - values["final_total_payment"]
    )

    return ((balance <= 0.01).astype(float) * 100).rename("metric_value")


def build_custom_metrics() -> dict[str, dict[str, Any]]:
    """Return the shared metrics used by all evaluation branch maps."""
    return {
        "outstanding_payment": {
            "name": "outstanding_payment",
            "label": "Average outstanding payment",
            "formula": calculate_outstanding_payment,
            "color_scheme": "Reds",
            "aggregation": "mean",
        },
        "paid_percentage": {
            "name": "paid_percentage",
            "label": "Paid cases (%)",
            "formula": calculate_paid_percentage,
            "color_scheme": [
                [0.0, "#b2182b"],
                [0.5, "#f4d03f"],
                [1.0, "#1a9850"],
            ],
            "aggregation": "mean",
            "color_min": 0,
            "color_max": 100,
        },
    }


def filter_dfg_by_frequency_coverage(dfg, coverage: float = 0.98):
    """Retain the most frequent edges until the requested coverage is met."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in the interval (0, 1].")
    sorted_edges = sorted(dfg.items(), key=lambda item: item[1], reverse=True)
    total = sum(frequency for _, frequency in sorted_edges)
    if total == 0:
        return {}, 0.0
    target = total * coverage
    filtered, cumulative = {}, 0
    for edge, frequency in sorted_edges:
        filtered[edge] = frequency
        cumulative += frequency
        if cumulative >= target:
            break
    return filtered, cumulative / total


def time_between_activities(
    event_log: pd.DataFrame,
    first_activity: str,
    second_activity: str,
) -> pd.DataFrame:
    """Notebook-compatible alias for first_later_activity_gap_days."""
    return first_later_activity_gap_days(event_log, first_activity, second_activity)


def cases_with_followed_by_relation(
    event_log: pd.DataFrame,
    first_activity: str,
    second_activity: str,
) -> set:
    """Return case IDs where the first activity occurs before a later second."""
    gaps = first_later_activity_gap_days(event_log, first_activity, second_activity)
    return set(gaps[PM_CASE_ID]) if PM_CASE_ID in gaps else set()


__all__ = [
    "PM_CASE_ID",
    "PM_ACTIVITY",
    "PM_TIMESTAMP",
    "build_evaluation_log",
    "add_variant_rank",
    "register_variant_stack",
    "register_without_payment",
    "add_reference_variant_rank",
    "register_reference_model_stack",
    "first_later_activity_gap_days",
    "add_delayed_notification_attributes",
    "register_delayed_notification_stack",
    "register_prefecture_stack",
    "register_judge_stack",
    "classify_nil_dismissal",
    "add_nil_dismissal_status",
    "evaluate_dismissal_comparison",
    "calculate_outstanding_payment",
    "calculate_paid_percentage",
    "build_custom_metrics",
    "filter_dfg_by_frequency_coverage",
    "time_between_activities",
    "cases_with_followed_by_relation",
]
