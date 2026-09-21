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
from typing import Union, Set
import pm4py
import pandas as pd


class EqToConstant(Predicate):

    def __init__(self, attribute_key: str, values: Union[str, Set[str]]):
        self.attribute_key = attribute_key
        self.values = sorted(list(values)) if isinstance(values, list) or isinstance(values, set) else [values]

    def evaluate(self, log: pd.DataFrame) -> pd.DataFrame:
        return pm4py.filter_event_attribute_values(log, self.attribute_key, self.values, level="case")

    def as_string(self) -> str:
        string_list = [ f"'{item}'" if isinstance(item, str) else str(item) for item in self.values]
        values_as_string = ', '.join(sorted(string_list))
        return f'({self.attribute_key} in {{ {values_as_string} }})'
