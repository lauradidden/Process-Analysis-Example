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
import pandas as pd
import copy


class LessThanConstant(Predicate):

    case_id_glue = 'case:concept:name'

    def __init__(self, attribute_key: str, value: float):
        self.attribute_key = attribute_key
        self.value = value

    def evaluate(self, log: pd.DataFrame) -> pd.DataFrame:
        filtered_df_by_ev = log[log[self.attribute_key] < self.value]
        i1 = log.set_index(LessThanConstant.case_id_glue).index
        i2 = filtered_df_by_ev.set_index(LessThanConstant.case_id_glue).index
        ret = log[i1.isin(i2)]
        ret.attrs = copy.copy(log.attrs) if hasattr(log, 'attrs') else {}
        return ret

    def as_string(self) -> str:
        return f'({self.attribute_key} < {self.value})'
