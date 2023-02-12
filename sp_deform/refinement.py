# -*- coding: utf-8 -*-
"""Реализация функций для работы с кривыми распределения зёрен по размерам и моделями измельчения
зёрен.
"""

from .curves import GrainsDistributionCurve
from .models import VolumeFractionWeightedMicrostructureState
from .specimen import Specimen
from .utils import get_optimal_time_step


def run_refinement_test(
    model,
    initial_distribution,
    strain_rate,
    max_strain,
    *,
    gauges=tuple(),
):
    """Создаёт экземпляр кривых, описывающих конечное распределение зёрен в образце с известным
    начальным распределением, испытанном на растяжение с указанной постоянной скоростью деформации.
    Дополнительно в ходе моделирования может заполнять переданные датчики.
    """
    if not isinstance(initial_distribution, GrainsDistributionCurve):
        raise TypeError(
            f'Initial distribution class is {initial_distribution.__class__.__name__},'
            ' while GrainsDistributionCurve is expected (do you forget to integrate?)'
        )
    initial_state = VolumeFractionWeightedMicrostructureState(initial_distribution.get_volumes())
    specimen = Specimen(model, initial_state)
    specimen.run_test(
        lambda time: time * strain_rate,
        lambda: specimen.strain >= max_strain,
        get_optimal_time_step(strain_rate),
        gauges=gauges,
    )
    return GrainsDistributionCurve(
        (
            (size, initial_distribution.cubes_sum * volume / (size ** 3))
            for size, volume in specimen.microstructure_state.get_volumes()
        ),
        cubes_sum=initial_distribution.cubes_sum,
    )
