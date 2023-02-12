# -*- coding: utf-8 -*-
"""Реализация классов и функций, предназначенных для получения кривых сверхпластичности из
результатов моделирования испытаний на одноосное растяжение образца с постоянной скоростью
деформации при известной модели материала.
"""

import logging

from .curves import RateCurve, StrainCurve
from .gauges import SpecimenGauge, MicrostructureGauge, LoadGauge, combine_gauges
from .utils import FrozenObject, ExtremaFinder, get_optimal_time_step


class RateCurveBuilder:
    """Базовый класс для алгоритмов, вычисляющих очередную точку скоростной кривой (зависимости
    напряжения от скорости деформации).
    """

    def __init__(self, strain_rate):
        self.strain_rate = strain_rate
        self.result = None

    def _run_test(self, specimen_factory, gauges):
        """Запускает испытание образца с заданной скоростью деформации. Аргумент `specimen_factory`
        должен быть вызываемым объектом, на каждый вызов возвращающим новый испытуемый образец
        (с нужной моделью и начальным состоянием микроструктуры материала).
        """
        specimen_factory().run_test(
            lambda time: self.strain_rate * time,
            self._end_condition,
            get_optimal_time_step(self.strain_rate),
            gauges,
        )

    def _end_condition(self):
        """Условие завершения испытания. Должно быть определено дочерним классом."""
        raise NotImplementedError


class MaxLoadRateCurveBuilder(RateCurveBuilder):
    """Строит точку скоростной кривой на основе положения максимума кривой нагрузки (инженерного
    напряжения).
    """

    def __init__(self, specimen_factory, strain_rate, **kwargs):
        """Заполняет базовые поля класса и запускает испытание образца."""
        super().__init__(strain_rate)
        self._max_load_finder = ExtremaFinder(extrema_type=ExtremaFinder.ExtremaType.SUP, **kwargs)

        self._strain_gauge = SpecimenGauge('strain')
        self._stress_gauge = MicrostructureGauge('stress')
        self._load_gauge = LoadGauge(self._stress_gauge)
        self._run_test(specimen_factory, (self._strain_gauge, self._stress_gauge, self._load_gauge))

    @property
    def max_load(self):
        return self._max_load_finder.extreme_value

    def _end_condition(self):
        cur_load = self._load_gauge.reading
        return self._max_load_finder.check_extrema_reached(cur_load, self._store_result)

    def _store_result(self, load):
        """Запоминает текущее положение как искомое положение максимума нагрузки."""
        self.result = FrozenObject(
            strain=self._strain_gauge.reading,
            stress=self._stress_gauge.reading,
        )


class PiecewiseRateCurveBuilder(RateCurveBuilder):
    """Строит точку скоростной кривой на основе идеи о том, что гладкая диаграмма деформирования
    является физической реализацией "идеальной" кусочно-линейной диаграммы, состоящей из начального
    (упругого) участка и участка пластического упрочнения. Определив наклон кривой на каждом из
    участков можно восстановить эту кусочно-линейную диаграмму и на пересечении составляющих её
    прямых (то есть, в точке перехода между участками) принять значение напряжения за пороговое.
    """
    ELASTIC_REGION_THRESHOLD = 0.9

    def __init__(
        self,
        specimen_factory,
        strain_rate,
        elastic_region_threshold=ELASTIC_REGION_THRESHOLD,
        **kwargs,
    ):
        """Заполняет базовые поля класса и запускает испытание образца."""
        super().__init__(strain_rate)
        self.initial_derivative = specimen_factory().model.normalized_young_modulus
        self.elastic_region_threshold = elastic_region_threshold
        self._elastic_region_passed = False

        self._prev_stress = self._prev_strain = None
        self._min_derivative_finder = ExtremaFinder(
            extrema_type=ExtremaFinder.ExtremaType.INF,
            **kwargs,
        )

        self._strain_gauge = SpecimenGauge('strain')
        self._stress_gauge = MicrostructureGauge('stress')
        self._run_test(specimen_factory, (self._strain_gauge, self._stress_gauge))

    @property
    def stabilized_derivative(self):
        return self._min_derivative_finder.extreme_value

    def _end_condition(self):
        """В начале ожидает уменьшения производной зависимости напряжения от деформации до доли
        `elastic_region_threshold` от её значения на упругом участке, которое принимается равным
        модулю Юнга материала. Далее ищет минимальное значение производной и использует его для
        построения искомой кусочно-линейной диаграммы.
        """
        cur_derivative = None
        if self._prev_stress is not None:
            cur_derivative = (
                (self._stress_gauge.reading - self._prev_stress)
                / (self._strain_gauge.reading - self._prev_strain)
            )
        self._prev_stress = self._stress_gauge.reading
        self._prev_strain = self._strain_gauge.reading

        if cur_derivative is None:
            return False
        if not self._elastic_region_passed:
            threshold = self.initial_derivative * self.elastic_region_threshold
            if cur_derivative > threshold:
                return False
            logging.debug(
                f'Elastic region passed at strain {self._prev_strain:.4};'
                f' derivative threshold {threshold:.1f}, got {cur_derivative:.1f}'
            )
            self._elastic_region_passed = True

        return self._min_derivative_finder.check_extrema_reached(cur_derivative, self._store_result)

    def _store_result(self, derivative):
        """Вычисляет и запоминает точку пересечения луча, описывающего упругий участок диаграммы
        деформирования, с касательной к диаграмме, построенной в её текущей точке.
        """
        intersection_strain = (
            (self._stress_gauge.reading - derivative * self._strain_gauge.reading)
            / (self.initial_derivative - derivative)
        )
        self.result = FrozenObject(
            strain=intersection_strain,
            stress=intersection_strain * self.initial_derivative,
        )


def build_rate_curve(builder_factory, strain_rates, specimen_factory):
    """Строит скоростную кривую, определяя точки с помощью указанного алгоритма (builder'а) для
    заданных скоростей деформации. Аргумент `specimen_factory` должен быть вызываемым объектом, на
    каждый вызов возвращающим новый испытуемый образец (с нужной моделью и начальным состоянием
    микроструктуры материала).
    """
    return RateCurve(
        (strain_rate, builder_factory(specimen_factory, strain_rate).result.stress)
        for strain_rate in sorted(strain_rates)
    )


def build_strain_curve(strain_rate, max_strain, specimen_factory):
    """Строит кривую деформирования до определённого значения при заданной скорости деформации.
    Аргумент `specimen_factory` должен быть вызываемым объектом, на каждый вызов возвращающим новый
    испытуемый образец (с нужной моделью и начальным состоянием микроструктуры материала).
    """
    strain_gauge = SpecimenGauge('strain')
    stress_gauge = MicrostructureGauge('stress')
    specimen_factory().run_test(
        lambda time: strain_rate * time,
        lambda: strain_gauge.reading >= max_strain,
        get_optimal_time_step(strain_rate),
        (strain_gauge, stress_gauge),
    )
    return StrainCurve(combine_gauges(strain_gauge, stress_gauge))


def build_grain_growth_curve(strain_rate, max_time, specimen_factory):
    """Строит кривую роста зёрен до определённого момента времени при заданной скорости деформации.
    Аргумент `specimen_factory` должен быть вызываемым объектом, на каждый вызов возвращающим новый
    испытуемый образец (с нужной моделью и начальным состоянием микроструктуры материала).
    """
    time_gauge = SpecimenGauge('time')
    size_gauge = MicrostructureGauge('grain_size')
    specimen_factory().run_test(
        lambda time: strain_rate * time,
        lambda: time_gauge.reading >= max_time,
        get_optimal_time_step(strain_rate),
        (time_gauge, size_gauge),
    )
    return combine_gauges(time_gauge, size_gauge)
