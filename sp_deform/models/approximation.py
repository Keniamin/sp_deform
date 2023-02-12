# -*- coding: utf8 -*-
"""Реализация аппроксимации заданных кривых сверхпластичности при помощи конкретной модели материала
и вспомогательных функций для неё.
"""

from ..curves import RateCurves, StrainCurves, GrainGrowthCurves
from ..curves_builders import (
    MaxLoadRateCurveBuilder,
    build_rate_curve, build_strain_curve, build_grain_growth_curve,
)
from ..specimen import Specimen
from ..utils import make_strain_rates
from .states import PlasticMicrostructureState


def make_specimen_factory(model, initial_grain_size):
    """Возвращает функцию, на каждый вызов создающую новый образец с указанной моделью материала и
    однородной микроструктурой с заданным размером зерна.
    """
    return lambda: Specimen(model, PlasticMicrostructureState(size=initial_grain_size))


def fit_curves_with_model(curves, model):
    """Приводит набор кривых в вид, пригодный для аппроксимации заданной моделью: преобразует в
    систему координат по умолчанию и нормирует кривые параметрами, соответствующими модели.
    """
    if hasattr(curves, 'transform'):
        curves = curves.transform()
    normalization = {
        attribute: getattr(model, attribute, None)
        for attribute in ('yield_stress', 'reference_rate')
        if hasattr(curves, attribute)
    }
    return curves.normalize(**normalization)


def approximate_curves(
    curves, model, *,
    rate_curves_divisor=(10 ** (1 / 3)),
    rate_curves_builder_factory=MaxLoadRateCurveBuilder,
):
    """Строит по модели набор кривых, соответствующий заданному. Нормировка совпадает с моделью."""
    raw_result = {}
    curves = fit_curves_with_model(curves, model)
    if isinstance(curves, RateCurves):
        for key, curve in curves:
            raw_result[key] = build_rate_curve(
                rate_curves_builder_factory,
                make_strain_rates(curve[0][0], curve[0][-1], divisor=rate_curves_divisor),
                make_specimen_factory(model, key),
            )
    elif isinstance(curves, StrainCurves):
        for key, curve in curves:
            raw_result[key.restore()] = build_strain_curve(
                key.rate,
                max(pt[0] for pt in curve),
                make_specimen_factory(model, key.size),
            )
    elif isinstance(curves, GrainGrowthCurves):
        for key, curve in curves:
            raw_result[key.restore()] = build_grain_growth_curve(
                key.rate,
                max(pt[0] for pt in curve),
                make_specimen_factory(model, key.size),
            )
    else:
        raise ValueError(f'Unknown curves type {curves.__class__.__name__}')
    return _make_normalized_curves(model, curves.__class__, data=raw_result)


def _make_normalized_curves(model, curves_class, *, data=None):
    """Создаёт набор кривых заданного типа, нормированных в соответствии с параметрами модели."""
    result = curves_class(data or {})
    for attribute in ('yield_stress', 'reference_rate'):
        if hasattr(result, attribute):
            setattr(result, attribute, getattr(model, attribute, None))
    return result
