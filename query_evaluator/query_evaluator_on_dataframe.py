"""
    This file is part of LogView.

    LogView is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LogView is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with LogView. If not, see <https://www.gnu.org/licenses/>.
"""

from logview.predicate.query import Query
from logview.interfaces import QueryEvaluator
import pandas as pd


class QueryEvaluatorOnDataFrame(QueryEvaluator):

    case_id_glue = 'case:concept:name'

    @staticmethod
    def _sanity_check(log: pd.DataFrame):
        expected_columns = ['case:concept:name', 'concept:name', 'time:timestamp', '@@index', '@@case_index']
        if not set(expected_columns).issubset(log.columns):
            raise RuntimeError("Not all expected columns are present. Did you read the log with pm4py?")

    def evaluate(self, log: pd.DataFrame, query: Query) -> (pd.DataFrame, pd.DataFrame):
        QueryEvaluatorOnDataFrame._sanity_check(log)
        filtered_log = query.evaluate(log)

        # get the complement of the result as all case_ids not present in filtered_log but present in log
        unique_values = filtered_log[QueryEvaluatorOnDataFrame.case_id_glue].unique()
        is_valid_case_id = log[QueryEvaluatorOnDataFrame.case_id_glue].isin(unique_values)
        return filtered_log, log[~is_valid_case_id]
