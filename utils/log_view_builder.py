from logview.log_view import LogView
from logview.query_evaluator import QueryEvaluatorOnDataFrame
from logview.query_registry import QueryRegistryImpl
import pandas as pd


class LogViewBuilder:
    @staticmethod
    def build_log_view(initial_source_log: pd.DataFrame) -> LogView:
        return LogView(QueryEvaluatorOnDataFrame(), QueryRegistryImpl(), initial_source_log)
