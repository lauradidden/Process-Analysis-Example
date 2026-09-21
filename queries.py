"""Reusable LogView queries for the ProMiSE P1 evaluation notebook.

This module only defines predicates. It does not build a LogView, modify the
event log, evaluate queries, or render filter-branch maps.

Some queries depend on case-level columns prepared in the notebook:

* ``variant_rank``
* ``reference_variant_rank``
* ``case_is_conforming``
* ``create_to_send_days``
* ``nil_dismissal_status``
"""

from logview.predicate import (
    EqToConstant,
    EventAttributeValues,
    EventuallyFollowsRelation,
    GreaterThanConstant,
    LessEqualToConstant,
    NotEqToConstant,
    Query,
)


PM_ACTIVITY = "concept:name"


# ---------------------------------------------------------------------------
# Variant Explorer: all variants -> first four -> first two
# ---------------------------------------------------------------------------

query_first_four_variants = Query(
    "FirstFourVariants",
    [LessEqualToConstant("variant_rank", 4)],
)

query_first_two_variants = Query(
    "FirstTwoVariants",
    [LessEqualToConstant("variant_rank", 2)],
)


# ---------------------------------------------------------------------------
# Dominant variants -> cases without Payment
# ---------------------------------------------------------------------------

query_without_payment = Query(
    "WithoutPayment",
    [NotEqToConstant(PM_ACTIVITY, "Payment")],
)


# ---------------------------------------------------------------------------
# Reference-model selection
# ---------------------------------------------------------------------------

query_without_credit_collection = Query(
    "WithoutCreditCollection",
    [NotEqToConstant(PM_ACTIVITY, "Send for Credit Collection")],
)

query_with_payment = Query(
    "WithPayment",
    [EventAttributeValues(PM_ACTIVITY, "Payment")],
)

query_first_six_reference_variants = Query(
    "FirstSixReferenceVariants",
    [LessEqualToConstant("reference_variant_rank", 6)],
)


# ---------------------------------------------------------------------------
# Delayed-notification investigation
# ---------------------------------------------------------------------------

query_nonconforming = Query(
    "NonConforming",
    [EqToConstant("case_is_conforming", "non-conforming")],
)

query_credit_collection_violation = Query(
    "CreditCollectionViolation",
    [EventAttributeValues(PM_ACTIVITY, "Send for Credit Collection")],
)

query_create_eventually_send = Query(
    "CreateFineEventuallySendFine",
    [EventuallyFollowsRelation([("Create Fine", "Send Fine")])],
)

query_more_than_87_days = Query(
    "CreateToSendMoreThan87Days",
    [GreaterThanConstant("create_to_send_days", 87)],
)


# ---------------------------------------------------------------------------
# Prefecture-appeal investigation
# ---------------------------------------------------------------------------

query_notification_to_prefecture_appeal = Query(
    "NotificationEventuallyPrefectureAppeal",
    [
        EventuallyFollowsRelation(
            [
                (
                    "Insert Fine Notification",
                    "Insert Date Appeal to Prefecture",
                )
            ]
        )
    ],
)

query_prefecture_dismissed = Query(
    "DismissedByPrefecture",
    [EqToConstant("dismissal", "#")],
)


# ---------------------------------------------------------------------------
# Judge-appeal investigation
# ---------------------------------------------------------------------------

query_notification_to_judge_appeal = Query(
    "NotificationEventuallyJudgeAppeal",
    [
        EventuallyFollowsRelation(
            [("Insert Fine Notification", "Appeal to Judge")]
        )
    ],
)

query_judge_dismissed = Query(
    "DismissedByJudge",
    [EqToConstant("dismissal", "G")],
)


# ---------------------------------------------------------------------------
# Credit-collection dismissal comparison
# ---------------------------------------------------------------------------

query_with_credit_collection = Query(
    "WithCreditCollection",
    [EventAttributeValues(PM_ACTIVITY, "Send for Credit Collection")],
)

query_dismissal_is_only_nil = Query(
    "DismissalIsOnlyNIL",
    [EqToConstant("nil_dismissal_status", "only_NIL")],
)

query_dismissal_is_g = Query(
    "DismissalIsG",
    [EqToConstant("dismissal", "G")],
)

query_dismissal_is_prefecture = Query(
    "DismissalIsPrefecture",
    [EqToConstant("dismissal", "#")],
)


__all__ = [
    "query_first_four_variants",
    "query_first_two_variants",
    "query_without_payment",
    "query_without_credit_collection",
    "query_with_payment",
    "query_first_six_reference_variants",
    "query_nonconforming",
    "query_credit_collection_violation",
    "query_create_eventually_send",
    "query_more_than_87_days",
    "query_notification_to_prefecture_appeal",
    "query_prefecture_dismissed",
    "query_notification_to_judge_appeal",
    "query_judge_dismissed",
    "query_with_credit_collection",
    "query_dismissal_is_only_nil",
    "query_dismissal_is_g",
    "query_dismissal_is_prefecture",
]
