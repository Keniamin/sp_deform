# -*- coding: utf-8 -*-
"""Реализация классов датчиков, фиксирующих состояние деформируемого образца и его микроструктуры в
процессе испытания.
"""

from .curves import LinearlyApproximatedCurve


class Gauge:
    """Базовый класс для датчиков общего назначения, хранящих историю измерений."""

    def __init__(self):
        """Создаёт датчик с пустой историей."""
        self._last_update=None
        self._data = []

    def __repr__(self):
        return f'<{self._name} with {len(self._data)} readings up to {self._last_update}>'

    @property
    def empty(self):
        """Возвращает признак отсутствия измеренных показаний."""
        return not self._data

    @property
    def reading(self):
        """Возвращает последнее измеренное показание."""
        if self._data:
            return self._data[-1]
        return None

    def store_reading(self, specimen):
        """Записывает текущее показание датчика в историю измерений."""
        self._data.append(self._measure(specimen))
        self._last_update = specimen.time

    @property
    def _name(self):
        return self.__class__.__name__

    def _measure(self, specimen):
        """Функция, вычисляющая текущее значение датчика по образцу."""
        raise NotImplementedError


class PropertyGauge(Gauge):
    """Базовый класс для датчиков, измеряющих некоторый именованный параметр."""

    def __init__(self, property, **kwargs):
        super().__init__(**kwargs)
        self.property = property

    @property
    def _name(self):
        return f'{self.__class__.__name__}({self.property})'


class DependentGauge(Gauge):
    """Базовый класс для зависимых датчиков, то есть вычисляющих свою величину на основе других."""

    def __init__(self, *, parent_gauges, **kwargs):
        """Сохраняет данные о родительских датчиках."""
        super().__init__(**kwargs)
        self.parent_gauges = parent_gauges

    def _measure(self, specimen):
        """Проверяет, что все родительские датчики имеют актуальное состояние и передаёт их текущие
        показания в функцию вычисления значения зависимого датчика.
        """
        parents_state = {}
        for name, gauge in self.parent_gauges.items():
            if gauge._last_update != specimen.time:
                raise RuntimeError(
                    f'Dependent gauge {self.__class__.__name__} read out at {specimen.time} before'
                    f' parent gauge {gauge}; check your gauges ordering'
                )
            parents_state[name] = gauge.reading
        return self._dependent_measure(specimen, **parents_state)

    def _dependent_measure(self, specimen, **kwargs):
        """Функция, вычисляющая текущее значение зависимого датчика по образцу и значениям
        родительских датчиков.
        """
        raise NotImplementedError


class CustomGauge(Gauge):
    """Класс, описывающий датчик, измеряющий произвольную заданную пользователем величину."""

    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    @property
    def _name(self):
        return f'{self.__class__.__name__} via {self.func.__name__}'

    def _measure(self, specimen):
        return self.func(specimen)


class SpecimenGauge(PropertyGauge):
    """Класс, описывающий датчик, измеряющий некоторый макроскопический параметр образца."""

    def _measure(self, specimen):
        return getattr(specimen, self.property)


class MicrostructureGauge(PropertyGauge):
    """Класс, описывающий датчик, измеряющий некоторый параметр микроструктуры образца."""

    def _measure(self, specimen):
        return specimen.microstructure_state.calc_property(self.property, specimen)


class LoadGauge(DependentGauge):
    """Класс, описывающий датчик, измеряющий нагрузку (инженерное напряжение) в образце."""

    def __init__(self, stress_gauge, **kwargs):
        parent_gauges = {'stress': stress_gauge}
        super().__init__(parent_gauges=parent_gauges, **kwargs)

    def _dependent_measure(self, specimen, stress):
        return stress / specimen.length


class DeformationEnergyGauge(DependentGauge):
    """Класс, описывающий датчик, измеряющий накопленную в образце энергию деформирования."""

    def __init__(self, stress_gauge, strain_gauge, **kwargs):
        super().__init__(
            parent_gauges={
                'stress': stress_gauge,
                'strain': strain_gauge,
            },
            **kwargs,
        )
        self.prev_stress = self.prev_strain = None
        self.energy = 0

    def _dependent_measure(self, specimen, stress, strain):
        """Интегрирование `stress` по `strain` методом трапеций."""
        if self.prev_stress is not None and self.prev_strain is not None:
            self.energy += 0.5 * (stress + self.prev_stress) * (strain - self.prev_strain)
        self.prev_stress, self.prev_strain = stress, strain
        return self.energy


def combine_gauges(gauge1, gauge2, data_scale=(1, 1)):
    """Собирает данные с двух датчиков в виде кривой, аппроксимирующей зависимость второй величины
    от первой. Опционально применяет к данным нормирующие коэффициенты.
    """
    if len(gauge1._data) != len(gauge2._data):
        raise RuntimeError(f'Failed to combine gauges: {gauge1} vs. {gauge2}')
    return LinearlyApproximatedCurve(zip(
        (value * data_scale[0] for value in gauge1._data),
        (value * data_scale[1] for value in gauge2._data),
    ))
