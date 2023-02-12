# -*- coding: utf-8 -*-
"""Реализация классов для моделей сверхпластичности, описывающих измельчение зёрен."""

import math

from ..utils import get_optimal_time_step
from .base_classes import SuperplasticityModel


class RefinementModel(SuperplasticityModel):
    """Базовый класс для моделей, реализующих алгоритмы измельчения и объединения зёрен."""
    parameters = {'combine_grain_groups_size_threshold': 0.05}

    def get_refined_groups(self, **kwargs):
        """Функция, реализующая модель измельчения зёрен. Возвращает `None`, если заданная входными
        параметрами группа зёрен не должна измельчаться, либо список пар "объёмная доля, размер
        зёрен, пластическая деформация" для новых групп.
        """
        raise NotImplementedError


class RateSensitivityRefinementModel(RefinementModel):
    """Базовый класс для моделей измельчения зёрен, реализующий критерий измельчения по работе
    `Ghosh A.K., Raj R. // Acta metallurgica, 1986`.
    """
    parameters = {
        'reference_refinement_size': 3,
        'reference_critical_rate': None,
        'mu': None,
    }

    def calc_critical_rate(self, size):
        """Вычисляет критическую (пороговую) скорость деформации для заданного размера зерна.
        Соответствует формуле (4) из указанной работы (с обобщённым степенным коэффициентом).
        """
        return self.reference_critical_rate * (self.reference_refinement_size / size) ** self.mu

    def calc_critical_size(self, rate):
        """Вычисляет критический размер зерна при заданной скорости деформации. Соответствует
        обращённой формуле (4) из указанной работы (с обобщённым степенным коэффициентом).
        """
        rev_mu = 1 / self.mu
        return self.reference_refinement_size * (self.reference_critical_rate / rate) ** rev_mu


class GhoshRajRefinementModel(RateSensitivityRefinementModel):
    """Модель измельчения зёрен по работе `Ghosh A.K., Raj R. // Acta metallurgica, 1986`."""
    parameters = {
        'b': 5,
        'epsilon_c': 4,
    }

    def get_refined_groups(
        self,
        *,
        full_strain,
        plastic_strain,
        grain_size,
        volume_fraction,
        **kwargs,
    ):
        strain_rate = self.get_plastic_strain_rate(
            full_strain=full_strain,
            plastic_strain=plastic_strain,
            grain_size=grain_size,
            volume_fraction=volume_fraction,
            **kwargs,
        )
        crit_rate = self.calc_critical_rate(grain_size)
        if strain_rate < crit_rate:
            return None

        q = 0.5 - math.atan(self.b * (1 - full_strain / self.epsilon_c)) / math.pi
        return (
            (q * volume_fraction, self.reference_refinement_size, full_strain),
            ((1 - q) * volume_fraction, math.sqrt(1 - q) * grain_size, plastic_strain),
        )


class GoncharovRefinementModel(RateSensitivityRefinementModel):
    """Модель измельчения по оригинальной диссертационной работе `Гончаров И.А. // 2021` с небольшой
    модификацией: коэффициент измельчения `r` обёрнут в нормированный арктангенс, чтобы изменить его
    ОДЗ с интервала `(0, 1)` на `(0, inf)`, который удобнее при поиске значений с помощью МНК.
    """
    parameters = {
        's_0': 0,
        'theta': 1,
        'r': 1,  # new grains size is 0.5 of original
    }

    def calc_refinement_probability(self, normalized_size):
        """Вычисляет вероятность измельчения зерна в единицу времени по его нормализованному размеру
        (то есть, отношению текущего размера к критическому для текущей скорости деформации).
        """
        return self.s_0 * normalized_size ** self.theta

    def calc_normalized_size(self, refinement_probability=1.0):
        """Вычисляет такой нормализованный размер зерна (то есть, отношение текущего размера к
        критическому для текущей скорости деформации), при котором вероятность измельчения зерна в
        единицу времени будет равна заданной.
        """
        try:
            return (refinement_probability / self.s_0) ** (1 / self.theta)
        except OverflowError:
            return math.inf

    def calc_terminal_size(self, strain_rate, time_step=None):
        """Вычисляет терминальный размер зерна для заданных скорости деформации и шага расчёта (то
        есть размер, при достижении которого зерно гарантированно измельчится на следующем шаге).
        """
        if time_step is None:
            time_step = get_optimal_time_step(strain_rate)
        norm_size = self.calc_normalized_size(1 / time_step)
        crit_size = self.calc_critical_size(strain_rate)
        return norm_size * crit_size

    def get_refined_groups(
        self,
        *,
        time_step,
        grain_size,
        plastic_strain,
        volume_fraction,
        **kwargs,
    ):
        strain_rate = self.get_plastic_strain_rate(
            time_step=time_step,
            grain_size=grain_size,
            plastic_strain=plastic_strain,
            volume_fraction=volume_fraction,
            **kwargs,
        )
        crit_size = self.calc_critical_size(strain_rate)
        if grain_size < crit_size:
            return None

        refinement_frac = time_step * self.calc_refinement_probability(grain_size / crit_size)
        if refinement_frac <= 0:
            return None
        if refinement_frac > 1:
            refinement_frac = 1

        size_coef = 2 * math.atan(self.r) / math.pi
        return (
            (refinement_frac * volume_fraction, size_coef * grain_size, plastic_strain),
            ((1 - refinement_frac) * volume_fraction, grain_size, plastic_strain),
        )
