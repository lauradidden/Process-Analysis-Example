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
from typing import Union, List
import pandas as pd


class Query (Predicate):

    def __init__(self, name: str, predicates: Union[Predicate, List[Predicate]]):
        self.name = name
        self.predicates = [predicates] if isinstance(predicates, Predicate) else predicates

    def evaluate(self, log: pd.DataFrame) -> pd.DataFrame:
        current_result = log
        for predicate in self.predicates:
            if len(current_result) == 0:
                break
            current_result = predicate.evaluate(current_result)
        return current_result

    def as_string(self) -> str:
        string_list = [predicate.as_string() for predicate in self.predicates]
        return ' and '.join(sorted(string_list))

    def get_name(self) -> str:
        return self.name
