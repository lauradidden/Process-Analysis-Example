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
import pm4py
import pandas as pd


class DurationWithin(Predicate):

    def __init__(self, min_duration_seconds: int, max_duration_seconds: int):
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds

    def evaluate(self, log: pd.DataFrame) -> pd.DataFrame:
        return pm4py.filter_case_performance(log, self.min_duration_seconds, self.max_duration_seconds)

    def as_string(self) -> str:
        return f'(DurationWithin [{self.min_duration_seconds}, {self.max_duration_seconds}])'
