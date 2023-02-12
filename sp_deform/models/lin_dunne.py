# -*- coding: utf-8 -*-
"""Реализация классов для моделей сверхпластичности из работ J. Lin, F.P.E. Dunne и соавторов."""

import math

from .base_classes import NormalizedModel


class HardeningModel(NormalizedModel):
    """Модель упрочнения по работе `Lin J., Dunne F.P.E., Hayhurst D.R. // Philosophical Magazine A,
    1996` в нормированном виде.
    """
    parameters = {'H_0': 0, 'H': 1}

    def get_hardening(self, *, plastic_strain, **kwargs):
        return self.H_0 * (1 - math.exp(-self.H * plastic_strain))


class PowerLawDeformationModel(NormalizedModel):
    """Модель деформирования по работе `Lin J., Dunne F.P.E., Hayhurst D.R. // Philosophical
    Magazine A, 1996` в нормированном виде.
    """
    parameters = {'B': 0, 'nu': 1, 'alpha': 1}

    def get_plastic_strain_rate(self, *, grain_size, **kwargs):
        plastic_stress = (
            + self.get_stress(grain_size=grain_size, **kwargs)
            - self.get_hardening(grain_size=grain_size, **kwargs)
            - 1
        )
        size_coef = 1 / grain_size ** self.alpha
        return size_coef * (self.B * max(plastic_stress, 0)) ** self.nu


class SinhDeformationModel(NormalizedModel):
    """Модель деформирования по работе `Zhou M., Dunne F.P.E. // The Journal of Strain Analysis for
    Engineering Design, 1996` в нормированном виде.
    """
    parameters = {'A': 0, 'B': 0, 'alpha': 1}

    def get_plastic_strain_rate(self, *, grain_size, **kwargs):
        plastic_stress = (
            + self.get_stress(grain_size=grain_size, **kwargs)
            - self.get_hardening(grain_size=grain_size, **kwargs)
            - 1
        )
        size_coef = self.A / grain_size ** self.alpha
        return size_coef * math.sinh(self.B * max(plastic_stress, 0))


class GrainGrowthModel(NormalizedModel):
    """Модель роста зёрен по работе `Cheong B.H., Lin J., Ball A.A. // Journal of Strain Analysis,
    2000` в нормированном виде.
    """
    parameters = {'D': 0, 'G': 0, 'beta': 1, 'phi': 1}

    def get_grain_size_rate(self, *, grain_size, **kwargs):
        plastic_strain_rate = self.get_plastic_strain_rate(grain_size=grain_size, **kwargs)
        return (
            + self.D / grain_size ** self.beta
            + self.G * plastic_strain_rate / grain_size ** self.phi
        )
