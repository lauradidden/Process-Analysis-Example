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

from logview.interfaces import Predicate
from typing import List
import pandas as pd


class Union(Predicate):

    case_id_glue = 'case:concept:name'

    def __init__(self, predicates: List[Predicate]):
        self.predicates = predicates

    def evaluate(self, log: pd.DataFrame) -> pd.DataFrame:
        result = set()
        for predicate in self.predicates:
            current_result = predicate.evaluate(log)
            unique_values = current_result[Union.case_id_glue].unique()
            result = result.union(unique_values)
        return log[log[Union.case_id_glue].isin(result)]

    def as_string(self) -> str:
        string_list = [predicate.as_string() for predicate in self.predicates]
        return ' or '.join(sorted(string_list))
