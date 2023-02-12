# -*- coding: utf-8 -*-
"""Реализация классов для моделей сверхпластичности из работ J. Lin, F.P.E. Dunne и соавторов."""

import math

from ..utils import COMPUTATIONAL_EPSILON
from .base_classes import SuperplasticityModel, NormalizedModel


class PerzinaHardeningModel(NormalizedModel):
    """Модель упрочнения, соответствующая модели Perzina в системе КЭ-моделирования ANSYS, в
    нормированном виде.
    """
    parameters = {'S_0': 0, 'H_0': 0, 'H': 1, 'H_1': 0}

    def get_hardening(self, *, plastic_strain, **kwargs):
        return (
            + self.S_0
            + self.H_0 * (1 - math.exp(-self.H * plastic_strain))
            + self.H_1 * plastic_strain
        )


class PerzinaDeformationModel(NormalizedModel):
    """Модель деформирования, соответствующая модели Perzina в системе КЭ-моделирования ANSYS, в
    нормированном виде.
    """
    parameters = {'A': 0, 'alpha': 1, 'nu': 1}

    def ensure_physical_ranges(self, policy):
        """Данная функция расширяет базовую (проверяющую на отрицательные значения) дополнительным
        ограничением. Показатель скоростной чувствительности `m` лежит в диапазоне от 0 до 1.
        Соответственно, параметр `nu = 1 / m` может принимать значения от 1 до бесконечности.
        """
        if self.nu < 1:
            if policy is SuperplasticityModel.PhysicalRangePolicy.IGNORE:
                pass
            elif policy is SuperplasticityModel.PhysicalRangePolicy.CUTOFF:
                self.nu = 1
            elif policy is SuperplasticityModel.PhysicalRangePolicy.INVERT:
                self.nu = 1 + (1 - self.nu)
            elif policy is SuperplasticityModel.PhysicalRangePolicy.RAISE:
                raise ValueError(
                    f'Attribute value nu={self.nu} for {self.__class__.__name__} object'
                    ' violates physical range'
                )
            else:
                raise ValueError(f'Unknown physical ranges policy {policy}')
        super().ensure_physical_ranges(policy)

    def get_plastic_strain_rate(self, *, grain_size, **kwargs):
        stress = self.get_stress(grain_size=grain_size, **kwargs)
        hardening = self.get_hardening(grain_size=grain_size, **kwargs)
        size_coef = self.A / grain_size ** self.alpha
        return size_coef * max(stress / max(hardening, COMPUTATIONAL_EPSILON) - 1, 0) ** self.nu
