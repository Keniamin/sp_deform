# -*- coding: utf-8 -*-
"""Реализация работы с искажёнными кривыми."""

import logging
import random

from copy import deepcopy


class DisturbanceModifier:
    """Класс, реализующий искажение ("зашумление") кривых путём наложения случайной погрешности,
    равномерно распределённой в пределах заданной относительной амплитуды.
    """

    def __init__(self, amplitude=0.05, patterns=None, *, generate_missed=True):
        """Позволяет задать амплитуду шума и заранее указать паттерны для кривых. Каждый паттерн
        должен быть набором чисел от -1 до 1 каждое в количестве, равном количеству точек в кривой.
        Числа задают относительную величину искажения ординаты соответствующей точки (в долях от
        максимальной амплитуды).
        По умолчанию при применении построенного объекта к кривой, паттерн для которой неизвестен,
        будет автоматически сгенерирован новый (случайный). Параметр `generate_missed=False`
        позволяет отключить это поведение и выбрасывать в таком случае исключение.
        """
        if patterns is None:
            self.patterns = {}
        else:
            self.patterns = deepcopy(patterns)
        self.generate_missed = generate_missed
        self.amplitude = amplitude

    def __call__(self, key, points):
        """Применяет к кривой заданный паттерн либо генерирует и применяет новый."""
        if key in self.patterns:
            pattern = self.patterns[key]
        elif self.generate_missed:
            pattern = self.patterns[key] = [
                round(random.uniform(-1, 1), 4)
                for _ in range(len(points))
            ]
            logging.info(f'Generated new pattern for {key} with length {len(pattern)}')
        else:
            raise RuntimeError(f'Pattern for {key} is missed but generating was denied')
        if len(pattern) != len(points):
            raise ValueError(
                f'Has pattern for {key} with length {len(pattern)}, but got {len(points)} points'
            )
        return (
            (pt[0], pt[1] * (1 + self.amplitude * relative_noise))
            for pt, relative_noise in zip(points, pattern)
        )
