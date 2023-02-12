# -*- coding: utf-8 -*-
"""Реализация базовых классов для наборов кривых (диаграмм деформирования) сверхпластичности."""

import os.path

from collections import namedtuple
from collections.abc import Iterable, Collection
from copy import copy
from enum import Enum

from ..utils import COMPUTATIONAL_EPSILON, DegreeFormat, load_data_file
from .curves import LinearlyApproximatedCurve


class SuperplasticityCurves(object):
    """Базовый класс для набора кривых сверхпластичности. Атрибуты:
        * `curve_class` задаёт класс, хранящий конкретную кривую. Класс не должен иметь внутреннего
        состояния, то есть должен полностью определяться данными, переданными в `__init__`.
        * `curve_key_view` задаёт класс, использующийся для "изображения" ключа кривой при переборе.
        Идея в том, что в самом наборе кривых ключ хранится в "сыром" виде, а такой промежуточный
        класс позволяет выдавать наружу более удобное для работы значение.
        * `names` задаёт пути к известным файлам кривых для их упрощённой загрузки по имени.
    """
    curve_class = LinearlyApproximatedCurve
    curve_key_view = lambda self, key: key
    names = {}

    def __init__(self, data_designator):
        """Создаёт набор кривых либо напрямую из данных, либо загружая данные из файла, заданного
        именем из списка известных путей или полным относительным или абсолютным путём.
        """
        if isinstance(data_designator, str):
            filepath = self.names.get(data_designator, data_designator)
            raw_data = load_data_file(os.path.join('curves', filepath))
        else:
            raw_data = data_designator

        def ensure_tuple(value):
            if isinstance(value, Iterable):
                return tuple(value)
            return value

        self.data = {
            ensure_tuple(key): self.curve_class(points)
            for key, points in raw_data.items()
        }

    def __repr__(self):
        keys = ', '.join(str(key) for key in sorted(self.data))
        return f'<{self.__class__.__name__} with keys {keys}>'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        for key in sorted(self.data):
            yield self.curve_key_view(key), self.data[key]

    def modify(self, modifier):
        """Возвращает новый набор кривых, в котором к кривым из исходного набора применена заданная
        функция фильтрации и/или преобразования точек кривой. Например:
        ```
            # убираем из каждой кривой последнюю точку
            curves.modify(lambda _, curve: curve[:-1])
            # сдвигаем все точки влево на единицу и делим значения на размер зёрен (ключ кривой)
            curves.modify(lambda size, curve: ((pt[0] - 1, pt[1] / size) for pt in curve))
        ```
        """
        result = copy(self)
        result.data = {
            key: self.curve_class(modifier(self.curve_key_view(key), curve))
            for key, curve in self.data.items()
        }
        return result

    def normalize(self):
        """Возвращает новый набор кривых, нормализованный соответствующими опорными значениями. По
        умолчанию кривые хранят обычные (размерные) величины. Базовый класс не знает про структуру
        хранящихся в нём кривых, поэтому просто возвращает их рекурсивную копию. Классы-наследники
        должны перехватывать известные им параметры нормализации, выполнять над полученной копией
        кривых необходимые преобразования и сохранять информацию о нормализующем значении,
        которое используется в конкретном наборе кривых.
        """
        result = copy(self)
        result.data = {
            key: self.curve_class(curve)
            for key, curve in self.data.items()
        }
        return result

    def filter(self, *patterns_groups):
        """Возвращает новый набор кривых, в котором кривые из исходного набора отфильтрованы по
        признаку соответствия ключа кривой хотя бы одному из шаблонов во каждой из заданных групп
        шаблонов. Классы-наследники могут перехватывать вызов функции и генерировать шаблоны из
        описания настроек фильтрации, заданных в более человекочитаемом виде (см. документацию к
        функции `_combine_filter_args`).
        """

        def check_pattern(value, pattern):
            """Проверка соответствия значения шаблону. Если значение является набором чисел —
            сопоставляет значение и шаблон поэлементно. `None` в шаблоне соответствует любому
            значению. Сравнение чисел производится нестрого, чтобы работать даже в случае ошибок
            округления при чтении данных из текстового файла.
            """
            if isinstance(value, Collection):
                if not isinstance(pattern, Collection) or len(pattern) != len(value):
                    raise ValueError(f'Bad pattern {pattern} for value {value}')
                return all(
                    check_pattern(the_value, the_pattern)
                    for the_value, the_pattern in zip(value, pattern)
                )
            if pattern is None:
                return True
            return abs(value - pattern) <= abs(value) * COMPUTATIONAL_EPSILON

        result = copy(self)
        result.data = {
            key: self.curve_class(curve)
            for key, curve in self.data.items()
            if all(
                any(check_pattern(key, pattern) for pattern in patterns)
                for patterns in patterns_groups
                if patterns
            )
        }
        return result

    @staticmethod
    def _combine_filter_args(name, single_val, multi_val):
        """Вспомогательная функция для создания паттернов из двух именованных аргументов. Таким
        образом можно задать фильтрацию как конкретного значения атрибута, так и набора значений.
        Совмещение нескольких пар атрибутов позволяет настраивать фильтрацию максимально гибко.
        Например, для кривых деформирования (ключ — размер зёрен и скорость деформации) любой из
        следующих фильтров будет валидным:
        ```
            # все кривые, относящиеся к конкретному размеру зёрен
            curves.filter(size=6.4)
            # две кривые по заданными размерам и скорости
            curves.filter(sizes=(6.4, 9.0), rate=1e-3)
            # две кривые с конкретными ключами
            curves.filter(keys=((6.4, 1e-3), (9.0, 1e-4)))
        ```
        А любой из следующих — невалидным:
        ```
            # размер должен быть числом, а не строкой
            curves.filter(size='6.4')
            # нельзя одновременно указывать и конкретное значение, и список
            curves.filter(size=6.4, sizes=(9.0, 11.5))
            # каждый ключ должен быть парой чисел
            curves.filter(keys=(6.4, 9.0))
            # напряжение не является частью ключа кривой деформирования
            curves.filter(stress=100)
        ```
        """
        if multi_val is not None:
            if single_val is not None:
                raise ValueError(f'Arguments "{name}" and "{name}s" must not be used together')
            if not isinstance(multi_val, Iterable):
                raise ValueError(f'"{name}s" argument value must be iterable')
            yield from multi_val
        elif single_val is not None:
            yield single_val


class SizeGroupedCurves(SuperplasticityCurves):
    """Базовый класс для набора кривых, сгруппированных по размеру зерна."""
    curve_key_view = lambda self, key: self.SizeKey(key)

    class SizeKey(float):
        """Класс, задающий ключ кривой и способ его форматирования (например, в легенде графика)."""

        def __str__(self):
            return f'd̃={float(self)}'

    def filter(self, *patterns_groups, size=None, sizes=None):
        size_patterns = self._combine_filter_args('size', size, sizes)
        return super().filter(tuple(size_patterns), *patterns_groups)


class SizeRateGroupedCurves(SuperplasticityCurves):
    """Базовый класс для набора кривых, сгруппированных по размеру зерна и скорости деформации."""
    curve_key_view = lambda self, key: self.SizeRateKey(*key, self.reference_rate or 1)

    class SizeRateKey(namedtuple('SizeRateKey', ('size', 'rate', 'reference_rate'))):
        """Класс, задающий ключ кривой и способ его форматирования (например, в легенде графика)."""

        def __str__(self):
            return f'd̃={self.size}, 𝜀̇={DegreeFormat(self.rate * self.reference_rate)}'

        def restore(self):
            """Возвращает оригинальный ключ кривой."""
            return (self.size, self.rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_rate = None

    def normalize(self, reference_rate=None, **kwargs):
        """Нормализует ключи кривых заданным значением скорости деформации. Передача `None` или
        вызов без параметров сбрасывает нормализацию до размерной величины (1/с).
        """
        result = super().normalize(**kwargs)
        result.reference_rate = reference_rate
        result.data = {
            (key[0], key[1] * (self.reference_rate or 1) / (reference_rate or 1)): points
            for key, points in result.data.items()
        }
        return result

    def filter(
        self,
        *patterns_groups,
        size=None, sizes=None,
        rate=None, rates=None,
        key=None, keys=None,
    ):
        size_patterns = self._combine_filter_args('size', size, sizes)
        rate_patterns = self._combine_filter_args('rate', rate, rates)
        key_patterns = self._combine_filter_args('key', key, keys)
        return super().filter(
            tuple((size, None) for size in size_patterns),
            tuple((None, rate) for rate in rate_patterns),
            tuple(key_patterns),
            *patterns_groups,
        )


class RateStrainGroupedCurves(SuperplasticityCurves):
    """Базовый класс для набора кривых, сгруппированных по скорости деформации и достигнутому в
    эксперименте значению деформации.
    """
    curve_key_view = lambda self, key: self.RateStrainKey(*key, self.reference_rate or 1)

    class RateStrainKey(namedtuple('RateStrainKey', ('rate', 'strain', 'reference_rate'))):
        """Класс, задающий ключ кривой и способ его форматирования (например, в легенде графика)."""

        def __str__(self):
            if self.strain == 0:
                return 'initial'
            return f'𝜀̇={DegreeFormat(self.rate * self.reference_rate)}, 𝜀={self.strain}'

        def restore(self):
            """Возвращает оригинальный ключ кривой."""
            return (self.rate, self.strain)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_rate = None

    @property
    def initial(self):
        return self.data[(0, 0)]

    def normalize(self, reference_rate=None, **kwargs):
        """Нормализует ключи кривых заданным значением скорости деформации. Передача `None` или
        вызов без параметров сбрасывает нормализацию до размерной величины (1/с).
        """
        result = super().normalize(**kwargs)
        result.reference_rate = reference_rate
        result.data = {
            (key[0] * (self.reference_rate or 1) / (reference_rate or 1), key[1]): points
            for key, points in result.data.items()
        }
        return result

    def filter(
        self,
        *patterns_groups,
        rate=None, rates=None,
        strain=None, strains=None,
        key=None, keys=None,
    ):
        rate_patterns = self._combine_filter_args('rate', rate, rates)
        strain_patterns = self._combine_filter_args('strain', strain, strains)
        key_patterns = self._combine_filter_args('key', key, keys)
        return super().filter(
            tuple((rate, None) for rate in rate_patterns),
            tuple((None, strain) for strain in strain_patterns),
            tuple(key_patterns),
            *patterns_groups,
        )


class RateBasedCurves(SuperplasticityCurves):
    """Базовый класс для набора кривых, описывающих зависимость некоторой величины от скорости
    деформации.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_rate = None

    def normalize(self, reference_rate=None, **kwargs):
        """Нормализует точки кривых заданным значением скорости деформации. Передача `None` или
        вызов без параметров сбрасывает нормализацию до размерной величины (1/с).
        """
        result = super().normalize(**kwargs)
        result.reference_rate = reference_rate
        result.data = {
            key: self.curve_class(
                (point[0] * (self.reference_rate or 1) / (reference_rate or 1), point[1])
                for point in points
            )
            for key, points in result.data.items()
        }
        return result


class StrainTimeBasedCurves(SizeRateGroupedCurves):
    """Базовый класс для набора кривых, сгруппированных по размеру зерна и скорости деформации и
    описывающих зависимость некоторой величины от деформации/времени.
    """

    class BaseVariable(Enum):
        """Задаёт величину, зависимость от которой описывает конкретный набор кривых."""
        STRAIN = 'strain'
        TIME = 'time'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_variable = self.default_base_variable

    def transform(self, *, base_variable=None):
        """Преобразует набор кривых, описывающих зависимость от одной из величин, в набор кривых,
        описывающих зависимость от другой. Для пересчёта между временем и деформацией использует
        скорость деформации, взятую из ключа кривой.
        """
        if base_variable is None:
            base_variable = self.default_base_variable
        if base_variable is self.base_variable:
            modifier = lambda key, points: points
        elif base_variable is self.BaseVariable.TIME:
            if self.base_variable is self.BaseVariable.STRAIN:
                modifier = lambda key, points: ((pt[0] / key.rate, pt[1]) for pt in points)
            else:
                raise NotImplementedError
        elif base_variable is self.BaseVariable.STRAIN:
            if self.base_variable is self.BaseVariable.TIME:
                modifier = lambda key, points: ((key.rate * pt[0], pt[1]) for pt in points)
            else:
                raise NotImplementedError
        else:
            raise ValueError(f'Unknown base variable {base_variable}')
        result = self.modify(modifier)
        result.base_variable = base_variable
        return result

    def normalize(self, reference_rate=None, **kwargs):
        """Перехватывает вызов `normalize` родительского класса, поскольку в режиме зависимости от
        времени необходимо нормализовать заданным значением скорости деформации сами точки кривых.
        """
        result = super().normalize(reference_rate=reference_rate, **kwargs)
        if result.base_variable is self.BaseVariable.TIME:
            result.data = {
                key: self.curve_class(
                    (point[0] * (reference_rate or 1) / (self.reference_rate or 1), point[1])
                    for point in points
                )
                for key, points in result.data.items()
            }
        return result


class StressValuedCurves(SuperplasticityCurves):
    """Базовый класс для набора кривых, описывающих зависимость напряжения от некоторой величины."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yield_stress = None

    def normalize(self, yield_stress=None, **kwargs):
        """Нормализует точки кривых заданным значением порогового напряжения. Передача `None` или
        вызов без параметров сбрасывает нормализацию до размерной величины (МПа).
        """
        result = super().normalize(**kwargs)
        result.yield_stress = yield_stress
        result.data = {
            key: self.curve_class(
                (point[0], point[1] * (self.yield_stress or 1) / (yield_stress or 1))
                for point in points
            )
            for key, points in result.data.items()
        }
        return result
