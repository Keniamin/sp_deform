# -*- coding: utf-8 -*-
"""Реализация различных вспомогательных классов и функций общего назначения."""

import logging
import math
import os.path

from ast import literal_eval
from enum import Enum

DATA_PATH = os.path.dirname(os.path.dirname(__file__))
COMPUTATIONAL_EPSILON = 1e-15
TYPICAL_STRAIN_RATE = 1e-3


class FrozenObject:
    """Класс, представляющий объект с фиксированным набором атрибутов, заданным при создании.
    Пользователь объекта может изменять эти атрибуты, но не может создавать произвольные новые.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setattr__(self, name, value):
        getattr(self, name)  # raises exception on unknown attribute
        super().__setattr__(name, value)

    def __repr__(self):
        attrs = ', '.join(f'{name}={value}' for name, value in self.__dict__)
        return f'{self.__class__.__name__}({attrs})'


class DegreeFormat:
    """Выводит число в "научном" формате: с показателем от 1 до `base` и степенью `base`. Например,
    для базы 10 (по умолчанию) переводит 0.0025 в 2.5×10⁻³.
    """
    SUPERSCRIPT_CHARS = dict(zip('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    __slots__ = ('value', 'base')

    def __init__(self, value, base=10):
        """Сохраняет значение и базу."""
        self.value = value
        self.base = base

    def __format__(self, format_spec):
        """Осуществляет непосредственно форматирование по запросу. К части с десятичным разделителем
        применяется указанная спецификация формата.
        """
        value, deg = self.value, 0
        while value >= self.base:
            value /= self.base
            deg += 1
        while 0 < value < 1:
            value *= self.base
            deg -= 1
        result = [value.__format__(format_spec)]
        if deg != 0:
            result.append(f'×{self.base}')
            if deg != 1:
                result.extend(self.SUPERSCRIPT_CHARS[ch] for ch in str(deg))
        return ''.join(result)


class ExtremaFinder:
    """Вспомогательный класс для обобщённого поиска экстремума функции, которая может как достигать
    его в определённой точке, так и асимптотически к нему стремиться.
    """
    STABILIZATION_THRESHOLD = 0.01  # 1%
    STABILIZATION_LENGTH = 25
    MISS_LENGTH = 5

    class ExtremaType(Enum):
        """Задаёт тип искомого экстремума."""
        INF = -1
        SUP = 1

    def __init__(
        self,
        extrema_type,
        stabilization_threshold=STABILIZATION_THRESHOLD,
        stabilization_length=STABILIZATION_LENGTH,
        miss_length=MISS_LENGTH,
    ):
        """Сохраняет тип искомого экстремума и параметры поиска, строит его начальное состояние."""
        self.extrema_type = extrema_type
        self.stabilization_threshold = stabilization_threshold
        self.stabilization_length = stabilization_length
        self.miss_length = miss_length

        self.extreme_value = None
        self._prev_value = None
        self._stabilized_length = 0
        self._missed_length = 0

    def check_extrema_reached(self, cur_value, candidate_callback=None):
        """Осуществляет поиск достаточно длинного "участка стабилизации", на котором относительное
        изменение значения на каждом шаге составляет не более, чем `stabilization_threshold`.
        Останавливается по достижению не менее, чем `stabilization_length` таких точек подряд, либо
        при нахождении не менее, чем `miss_length` точек, на протяжении которых значения стремятся
        в противоположную от искомой сторону (например, стабильно убывают при поиске максимума).
        Опционально вызывает функцию `candidate_callback` при нахождении каждого нового "кандидата"
        на экстремум.
        """
        if (
            self.extreme_value is None
            or self.extrema_type.value * (cur_value - self.extreme_value) > 0
        ):
            if candidate_callback is not None:
                candidate_callback(cur_value)
            self.extreme_value = cur_value
        if self._prev_value is not None:
            values_diff = cur_value - self._prev_value
            if self.extrema_type.value * values_diff < 0:
                self._missed_length += 1
            else:
                self._missed_length = 0
            if abs(values_diff) < abs(self._prev_value) * self.stabilization_threshold:
                self._stabilized_length += 1
            else:
                self._stabilized_length = 0
        self._prev_value = cur_value
        if (
            self._stabilized_length >= self.stabilization_length
            or self._missed_length >= self.miss_length
        ):
            logging.debug(
                f'Found extrema with {self._stabilized_length} of {self.stabilization_length}'
                f' stable points and {self._missed_length} of {self.miss_length} missed points'
            )
            return True
        return False


def get_optimal_time_step(strain_rate):
    """Эмпирическая формула, дающая разумную величину шага по времени для испытания образца — чтобы
    на результирующих кривых деформирования получалось достаточно много точек для пренебрежения
    ошибкой линейного приближения между ними, но и расчёт при этом не занимал катастрофически
    большое время.
    """
    return TYPICAL_STRAIN_RATE / (strain_rate or TYPICAL_STRAIN_RATE) ** 0.5


def make_strain_rates(min_strain_rate, max_strain_rate, *, divisor):
    """Выдаёт список скоростей деформации, лежащих между указанными границами и являющихся при этом
    целой степенью заданного числа.
    """
    if divisor < 1 + COMPUTATIONAL_EPSILON:
        raise ValueError(f'Strain rate divisor must be larger than 1')
    result = []
    # Moving from end to beginning to avoid complicated code dealing with rounding errors
    degree = math.log(max_strain_rate) // math.log(divisor)
    while divisor ** degree >= min_strain_rate:
        result.append(divisor ** degree)
        degree -= 1
    # Ensure bounding values themselves are presented in the result
    if abs(min_strain_rate - result[-1]) > min_strain_rate * COMPUTATIONAL_EPSILON:
        result.append(min_strain_rate)
    if abs(max_strain_rate - result[0]) > max_strain_rate * COMPUTATIONAL_EPSILON:
        result.insert(0, max_strain_rate)
    return list(reversed(result))


def load_data_file(path):
    """Загружает файл по пути — либо абсолютному, либо относительно корня пакета."""
    with open(os.path.join(DATA_PATH, path), encoding='utf-8') as inp:
        return literal_eval(inp.read())


def make_common_layout(*, add_xaxis=False, add_yaxis=False):
    """Создаёт стандартные настройки для графиков, опционально добавляет параметры для осей."""
    layout = dict(
        font=dict(
            family='Liberation Serif',
            size=14,
        ),
        separators='.`',
    )

    def _make_axis_layout():
        return dict(
            exponentformat='none',
            rangemode='normal',
            title=dict(),
        )

    if add_xaxis:
        layout['xaxis'] = _make_axis_layout()
    if add_yaxis:
        layout['yaxis'] = _make_axis_layout()
    return layout


def update_with_defaults(params, **defaults):
    """Добавляет к словарю значения по умолчанию."""
    if params is not None:
        defaults.update(params)
    return defaults
