# -*- coding: utf-8 -*-
"""Реализация классов для наборов конкретных кривых (диаграмм деформирования) сверхпластичности."""

import logging
import math

from collections import defaultdict
from enum import Enum

from .base_classes import (
    SizeGroupedCurves, RateStrainGroupedCurves,
    RateBasedCurves, StrainTimeBasedCurves,
    StressValuedCurves,
)
from .curves import (
    RateSensitivityCurve, RateCurve,
    StrainCurve, GrainGrowthCurve,
    ContinuousGrainsDistributionCurve,
)


class RateSensitivityCurves(RateBasedCurves, SizeGroupedCurves):
    """Набор кривых, описывающих скоростную чувствительность материала."""
    curve_class = RateSensitivityCurve


class RateCurves(StressValuedCurves, RateBasedCurves, SizeGroupedCurves):
    """Набор кривых, описывающих зависимость напряжения течения материала от скорости деформации."""
    curve_class = RateCurve
    names = {
        '7475al_790K_ham': 'rates/Ghosh, Hamilton (7475 Al, 516C).py',
        '7475al_790K_raj': 'rates/Ghosh, Raj (7475 Al, 790K).py',
        'ti6al4v_1173K_imsp': 'rates/Kruglov et al (Ti-6Al-4V, 900C).py',
        'ti6al4v_1173K_misis': 'rates/Vargin, Burhanov, Zung, Polkin (Ti-6Al-4V, 900C).py',
        'ti6al4v_1200K': 'rates/Ghosh, Hamilton (Ti-6Al-4V, 927C).py',
    }

    def make_rate_sensitivity_curves(self):
        """Создаёт набор кривых скоростной чувствительности материала на основе набора скоростных
        кривых. Сохраняет информацию о нормализации.
        """
        result = RateSensitivityCurves({
            size: curve.make_rate_sensitivity_curve(_raw=True)
            for size, curve in self
        })
        result.reference_rate = self.reference_rate
        return result


class StrainCurves(StressValuedCurves, StrainTimeBasedCurves):
    """Набор кривых, описывающих зависимость напряжения в материале от деформации или времени."""
    default_base_variable = StrainTimeBasedCurves.BaseVariable.STRAIN
    curve_class = StrainCurve
    names = {
        '7475al_790K': 'strain/Ghosh, Raj (7475 Al, 516C).py',
        'alznmg_788K': 'strain/Dunne (Al-Zn-Mg, 788K).py',
        'ti6al4v_1173K': 'strain/Dunne+Enikeev (Ti-6Al-4V, 900C).py',
        'ti6al4v_1200K': 'strain/Ghosh, Hamilton (Ti-6Al-4V, 927C).py',
    }

    def make_rate_curves(self):
        """Создаёт набор скоростных кривых на основе набора кривых деформирования с помощью поиска
        точек максимума нагрузки (инженерного напряжения). Сохраняет информацию о нормализации.
        """
        rate_curves_data = defaultdict(list)
        for key, curve in self.transform():
            prev_stress = prev_load = None
            for strain, stress in curve:
                cur_load = stress / math.exp(strain)
                if prev_load is not None and cur_load < prev_load:
                    break
                prev_stress, prev_load = stress, cur_load
            rate_curves_data[key.size].append((key.rate, prev_stress))
        result = RateCurves(rate_curves_data)
        result.reference_rate = self.reference_rate
        result.yield_stress = self.yield_stress
        return result

    def extract_exact(self, abscissas, *, strict=False, **kwargs):
        """Извлекает из каждой кривой набора точки по запрошенным абсциссам (деформации или времени
        в зависимости от режима). По умолчанию абсциссы вне диапазона каждой кривой отбрасываются.
        Задание параметра `strict=True` изменяет это поведение на выбрасывание исключения, если с
        какой-либо кривой невозможно получить в точности запрошенные абсциссы.
        """

        def extractor(key, curve):
            if strict:
                curve_abscissas = abscissas
            else:
                curve_abscissas = [
                    abscissa for abscissa in abscissas
                    if curve[0][0] <= abscissa <= curve[-1][0]
                ]
                skipped = len(abscissas) - len(curve_abscissas)
                if skipped > 0:
                    logging.info(
                        f'Skipped {skipped} out-of-bounds abscissas while extracting points for {key}'
                    )
            return curve.extract_exact(curve_abscissas, **kwargs, _raw=True)

        return self.modify(extractor)

    def extract_uniform(self, *args, **kwargs):
        """Извлекает из каждой кривой набора указанное количество точек с абсциссами (деформацией
        или временем в зависимости от режима), равномерно распределёнными от нуля до максимальной
        на кривой.
        """
        return self.modify(lambda _, curve: curve.extract_uniform(*args, **kwargs, _raw=True))

    def extract_log(self, *args, **kwargs):
        """Извлекает из каждой кривой набора точки с абсциссами (деформацией или временем в
        зависимости от режима), логарифмически распределёнными от максимальной на кривой до
        заданной минимальной.
        """
        return self.modify(lambda _, curve: curve.extract_log(*args, **kwargs, _raw=True))


class GrainGrowthCurves(StrainTimeBasedCurves):
    """Набор кривых, описывающих зависимость размера зерна в материале от деформации или времени."""
    curve_class = GrainGrowthCurve
    default_base_variable = StrainTimeBasedCurves.BaseVariable.TIME
    names = {
        '7475al_790K': 'grains/Ghosh, Raj (7475 Al, 516C).py',
        'alznmg_788K': 'grains/Dunne (Al-Zn-Mg, 788K).py',
        'ti6al4v_1173K': 'grains/Dunne (Ti-6Al-4V, 900C).py',
        'ti6al4v_1200K': 'grains/Ghosh, Hamilton (Ti-6Al-4V, 927C).py',
    }


class ContinuousGrainsDistributionCurves(RateStrainGroupedCurves):
    """Набор кривых, описывающих распределение зёрен по размерам в виде непрерывной зависимости
    количества зёрен от их размера.
    """
    curve_class = ContinuousGrainsDistributionCurve
    names = {
        'ti6al4v_1173K': 'volumes/Dunne (Ti-6Al-4V, 900C).py',
    }
