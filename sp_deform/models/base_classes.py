# -*- coding: utf-8 -*-
"""Реализация базовых классов для моделирования сверхпластического деформирования."""

import logging

from enum import Enum

from ..utils import FrozenObject


class SuperplasticityModel(FrozenObject):
    """Базовый класс для моделей сверхпластичности с фиксированным набором феноменологических
    параметров.
    """

    class PhysicalRangePolicy(Enum):
        """Политика обработки значений, выпадающих за допустимый физический диапазон (например,
        множитель меньше нуля). Такие значения можно игнорировать (продолжать расчёт как есть),
        обрезать до ближайшего допустимого значения, инвертировать относительно границ диапазона
        (например, брать по модулю), либо выбрасывать исключение при любом их возникновении.
        """
        IGNORE = 'ignore'
        CUTOFF = 'cutoff'
        INVERT = 'invert'
        RAISE = 'raise'

    def __init__(self, **kwargs):
        """Создаёт объект-модель по известным значениям параметров.
        Предупреждает о "лишних" (неизвестных) параметрах.
        """
        params = self._collect_parameters()
        for name in params:
            if name in kwargs:
                params[name] = kwargs.pop(name)
            if params[name] is None:
                raise ValueError(
                    f'Attribute "{name}" is required for {self.__class__.__name__} object'
                )
        if kwargs:
            unexpected_attrs = '", "'.join(kwargs)
            logging.warning(
                f'Attributes "{unexpected_attrs}" was ignored for {self.__class__.__name__} object'
            )
        super().__init__(**params)

    def ensure_physical_ranges(self, policy):
        """Проверяет значения модели на соответствие физически допустимому диапазону в соответствии
        с заданной политикой. Базовая функция проверяет значения на отрицательность, дополнительная
        логика может быть реализована в классах-наследниках (например, для значений от 0 до 1).
        """
        for name in self._collect_parameters():
            value = getattr(self, name)
            if value < 0:
                if policy is SuperplasticityModel.PhysicalRangePolicy.IGNORE:
                    pass
                elif policy is SuperplasticityModel.PhysicalRangePolicy.CUTOFF:
                    setattr(self, name, 0)
                elif policy is SuperplasticityModel.PhysicalRangePolicy.INVERT:
                    setattr(self, name, -value)
                elif policy is SuperplasticityModel.PhysicalRangePolicy.RAISE:
                    raise ValueError(
                        f'Attribute value {name}={value} for {self.__class__.__name__} object'
                        ' violates physical range'
                    )
                else:
                    raise ValueError(f'Unknown physical range policy {policy}')

    def to_dict(self):
        """Возвращает словарь со значениями существенных (отличных от значений по умолчанию)
        параметров модели. Соблюдает инвариант `model.__class__(**model.to_dict()) == model`.
        """
        result = {}
        for name, default in self._collect_parameters().items():
            value = getattr(self, name)
            if value != default:
                result[name] = value
        return result

    def __repr__(self):
        essential_params = ', '.join(f'{name}={value}' for name, value in self.to_dict().items())
        return f'{self.__class__.__name__}({essential_params})'

    @classmethod
    def _collect_parameters(cls):
        """Собирает из классов-наследников значения атрибута `parameters` и формирует из них общий
        набор параметров со значениями по умолчанию.
        """
        parents = list(cls.__mro__)
        while parents.pop() is not SuperplasticityModel:
            pass  # remove base classes
        result = {}
        for parent in reversed(parents):
            result.update(getattr(parent, 'parameters', {}))
        return result


class NoHardeningModel(SuperplasticityModel):
    """Класс для моделей, не учитывающих упрочнение."""

    def get_hardening(self, **kwargs):
        return 0


class NoGrainGrowthModel(SuperplasticityModel):
    """Класс для моделей, не учитывающих рост зёрен."""

    def get_grain_size_rate(self, **kwargs):
        return 0


class NormalizedModel(SuperplasticityModel):
    """Базовый класс для моделей, нормированных опорной скоростью деформации и значением порогового
    напряжения.
    """
    parameters = {
        'reference_rate': None,
        'yield_stress': None,
    }


class ElasticStressModel(NormalizedModel):
    """Класс для моделей, представляющих деформацию в виде суммы упругой и пластической компонент и
    вычисляющих напряжение по упругой деформации.
    """
    parameters = {'normalized_young_modulus': None}

    def get_stress(self, *, full_strain, plastic_strain, **kwargs):
        return self.normalized_young_modulus * (full_strain - plastic_strain)
