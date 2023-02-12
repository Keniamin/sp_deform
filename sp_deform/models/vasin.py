# -*- coding: utf-8 -*-
"""Реализация классов для моделей сверхпластичности из работ Р.А. Васина и соавторов."""

from .base_classes import NormalizedModel


class OriginalDeformationModel(NormalizedModel):
    """Модель деформирования по работе `Быля О.И., Васин Р.А. // Известия Тульского государственного
    университета. Естественные науки, 2011` со степенным параметром вместо фиксированной степени.
    """
    parameters = {'L': 0, 'A': 0, 'B': 0, 'alpha': 2}

    def _get_rate(self, grain_size, stress, plastic_stress):
        return (
            + self.L * stress
            + self.A * plastic_stress ** 2 / grain_size ** self.alpha
            + self.B * plastic_stress ** 4
        )

    def get_plastic_strain_rate(self, *, grain_size, **kwargs):
        stress = self.get_stress(grain_size=grain_size, **kwargs)
        return self._get_rate(grain_size, stress, plastic_stress=stress)


class PlasticDeformationModel(OriginalDeformationModel):
    """Модифицированная модель деформирования с добавлением порогового напряжения."""

    def get_plastic_strain_rate(self, *, grain_size, **kwargs):
        stress = self.get_stress(grain_size=grain_size, **kwargs)
        return self._get_rate(grain_size, stress, plastic_stress=max(0, stress - 1))


class GrainGrowthModel(NormalizedModel):
    """Модель роста зёрен по работе `Быля О.И., Васин Р.А. // Известия Тульского государственного
    университета. Естественные науки, 2011`.
    """
    parameters = {'C_1': 0, 'C_2': 0, 'C_3': 0}

    def get_grain_size_rate(self, *, grain_size, **kwargs):
        stress = self.get_stress(grain_size=grain_size, **kwargs)
        numerator = (self.C_1 * stress ** 4 + self.C_2 * stress ** 5)
        return numerator / (1 + self.C_3 * grain_size * stress ** 4)
