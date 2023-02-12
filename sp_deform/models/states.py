# -*- coding: utf-8 -*-
"""Реализация классов, описывающих микроструктуру в процессе сверхпластического деформирования."""

import logging

from collections import namedtuple

from scipy.integrate import RK45

from ..curves import SortedCurve
from ..utils import COMPUTATIONAL_EPSILON


class PlasticMicrostructureState:
    """Класс, описывающий состояние микроструктуры материала в виде пары чисел: накопленной
    пластической деформации и среднего размера зёрен.
    """
    __slots__ = ('strain', 'size')

    def __init__(self, *, strain=0, size):
        self.strain = strain
        self.size = size

    def __str__(self):
        return f'd={self.size}, 𝜀ₚ={self.strain}'

    def make_step(self, specimen, time_step):
        """Вычисляет состояние микроструктуры по прошествии времени `time_step` путём интегрирования
        системы определяющих соотношений материала в соответствии с программой деформации образца.
        """

        def get_derivative(time, state):
            model_state = dict(
                time=time,
                full_strain=specimen.get_strain(time),
                plastic_strain=state[0],
                grain_size=state[1],
            )
            return (
                specimen.model.get_plastic_strain_rate(**model_state),
                specimen.model.get_grain_size_rate(**model_state),
            )

        self.strain, self.size = _integrate_via_runge_kutta(
            get_derivative,
            specimen.time,
            (self.strain, self.size),
            time_step,
        )

    def calc_property(self, property, specimen):
        """Вычисляет по текущему состоянию микроструктуры и образца запрошенную характеристику
        микроструктуры в соответствии с моделью материала.
        """
        model_state = {
            'time': specimen.time,
            'full_strain': specimen.strain,
            'plastic_strain': self.strain,
            'grain_size': self.size,
        }
        if property in model_state:
            return model_state[property]
        calculator = getattr(specimen.model, f'get_{property}')
        return calculator(**model_state)


class VolumeFractionWeightedMicrostructureState:
    """Класс, описывающий состояние микроструктуры материала в виде распределения групп зёрен,
    каждая из которых занимает определённую долю от общего объёма образца и содержит зёрна
    определённого размера с некоторой накопленной пластической деформацией. Объёмные доли групп в
    сумме должны давать 1, однако группы с долей меньше `MIN_TRACKABLE_FRACTION` отбрасываются для
    ускорения расчёта (что может незначительно уменьшать общую сумму).
    """
    MIN_TRACKABLE_FRACTION = 1e-5  # 0.001% of total volume
    __slots__ = ('grain_groups',)

    CombineTarget = namedtuple('CombineTarget', ('group', 'original_size', 'size_threshold'))

    class GrainGroup:
        """Группа зёрен, занимающая определённую долю объёма и внутреннее состояние."""

        def __init__(self, volume, state):
            self.volume = volume
            self.state = state

    def __init__(self, distribution):
        """Задаёт состояние микроструктуры распределением объёмных долей по размеру."""
        total_volume = sum(volume for _, volume in distribution)
        if abs(total_volume - 1) > self.MIN_TRACKABLE_FRACTION:
            raise ValueError(f'Total volume of the specimen must be 1, got: {total_volume}')
        self.grain_groups = [
            self.GrainGroup(volume=volume, state=PlasticMicrostructureState(size=size))
            for size, volume in distribution
        ]

    def make_step(self, specimen, time_step):
        """Вычисляет состояние микроструктуры по прошествии времени `time_step` путём вычисления
        состояния каждой группы и применения процедур измельчения зёрен и/или склейки групп с
        близкими параметрами (в зависимости от настроек модели).
        """
        for group in self.grain_groups:
            group.state.make_step(specimen, time_step)

        if getattr(specimen.model, 'get_refined_groups', None) is not None:
            self._refine_groups(specimen, time_step)

        if getattr(specimen.model, 'combine_grain_groups_size_threshold', None) is not None:
            self._combine_groups(specimen, time_step)

    def calc_property(self, property, specimen):
        """Вычисляет по текущему состоянию микроструктуры и образца запрошенную характеристику
        микроструктуры в соответствии с моделью материала — как взвешенное по объёму среднее той же
        характеристики по всем группам зёрен.
        """
        return sum(
            group.volume * group.state.calc_property(property, specimen)
            for group in self.grain_groups
        )

    def get_volumes(self):
        """Возвращает текущее распределение объёмных долей групп зёрен по размерам."""
        return SortedCurve((group.state.size, group.volume) for group in self.grain_groups)

    def _refine_groups(self, specimen, time_step):
        """Реализует процедуру измельчения зёрен, вызывая соответствующую функцию модели."""
        new_grain_groups = []
        refinement_time = specimen.time + time_step
        for group in self.grain_groups:
            refinement_result = specimen.model.get_refined_groups(
                time=refinement_time,
                time_step=time_step,
                full_strain=specimen.get_strain(refinement_time),
                plastic_strain=group.state.strain,
                grain_size=group.state.size,
                volume_fraction=group.volume,
            )
            if refinement_result is None:
                new_grain_groups.append(group)
                continue

            groups_metainfo = []
            for volume, size, strain in refinement_result:
                new_group = self.GrainGroup(
                    state=PlasticMicrostructureState(size=size, strain=strain),
                    volume=volume,
                )
                new_grain_groups.append(new_group)
                groups_metainfo.append(
                    f'  - 0x{id(new_group):X} ({new_group.state}, v={new_group.volume:.6})'
                )

            groups_metainfo = '\n'.join(groups_metainfo)
            logging.debug(
                f'For specimen 0x{id(specimen):X} after time step {specimen.time:.3}+{time_step:6}'
                f' grain group 0x{id(group):X} ({group.state}, v={group.volume:.6}) refined into'
                f' {len(refinement_result)} groups:\n{groups_metainfo}'
            )
        self.grain_groups = new_grain_groups

    def _combine_groups(self, specimen, time_step):
        """Реализует алгоритм склейки групп зёрен с близкими параметрами, описанный в оригинальной
        работе. Порог близости задаётся параметром `combine_grain_groups_size_threshold` у модели.
        """
        time_step_str = f'time step {specimen.time:.3}+{time_step:6}'
        new_grain_groups, combine_target = [], None
        for group in sorted(self.grain_groups, key=lambda group: group.state.size):
            if group.volume < self.MIN_TRACKABLE_FRACTION:
                logging.debug(
                    f'For specimen 0x{id(specimen):X} after {time_step_str}'
                    f' eliminated grain group 0x{id(group):X} ({group.state}, v={group.volume:.6})'
                )
                continue
            if combine_target is not None and group.state.size > combine_target.size_threshold:
                new_grain_groups.append(combine_target.group)
                combine_target = None
            if combine_target is None:
                threshold = group.state.size + specimen.model.combine_grain_groups_size_threshold
                combine_target = self.CombineTarget(
                    group=group,
                    original_size=group.state.size,
                    size_threshold=threshold,
                )
                continue

            target_group = combine_target.group
            logging.debug(
                f'For specimen 0x{id(specimen):X} after {time_step_str}'
                f' grain group 0x{id(group):X} ({group.state}, v={group.volume:.6}) combined with'
                f' group 0x{id(target_group):X} ({target_group.state}, v={target_group.volume:.6});'
                f' original sizes difference was {group.state.size - combine_target.original_size}'
            )
            new_volume = target_group.volume + group.volume
            for attribute in PlasticMicrostructureState.__slots__:
                weighted_sum = (
                    target_group.volume * getattr(target_group.state, attribute)
                    + group.volume * getattr(group.state, attribute)
                )
                setattr(target_group.state, attribute, weighted_sum / new_volume)
            target_group.volume = new_volume
        if combine_target is not None:
            new_grain_groups.append(combine_target.group)
        self.grain_groups = new_grain_groups


def _integrate_via_runge_kutta(get_derivative, initial_time, initial_state, time_step):
    """Интегрирует систему дифференциальных уравнений методом Рунге-Кутты с проверкой ошибок."""
    target_time = initial_time + time_step
    rk = RK45(
        get_derivative,
        t0=initial_time,
        y0=initial_state,
        t_bound=target_time,
        first_step=(0.5 * time_step),
    )
    while rk.status == 'running':
        message = rk.step()
    if rk.status != 'finished':
        raise RuntimeError(f'Runge-Kutta step failed: {message}')
    if abs(target_time - rk.t) > abs(target_time) * COMPUTATIONAL_EPSILON:
        raise RuntimeError('Runge-Kutta integral time mismatch')
    return rk.y
