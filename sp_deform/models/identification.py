# -*- coding: utf8 -*-
"""Реализация алгоритма поиска значений для параметров (идентификации) моделей и вспомогательных
функций для него.
"""

import logging
import math

from scipy.optimize import leastsq

from ..curves import RateCurves, StrainCurves, GrainGrowthCurves
from ..curves_builders import (
    MaxLoadRateCurveBuilder,
    build_rate_curve, build_strain_curve, build_grain_growth_curve,
)
from .base_classes import SuperplasticityModel
from .approximation import make_specimen_factory, fit_curves_with_model

LEVEL_PROGRESS = (logging.DEBUG + logging.INFO) // 2

logging.addLevelName(LEVEL_PROGRESS, 'PROGRESS')


def find_optimal_params(
    model,
    params,
    functional,
    *,
    physical_range_policy=SuperplasticityModel.PhysicalRangePolicy.INVERT,
    **kwargs,
):
    """С помощью метода наименьших квадратов подбирает такие значения запрошенных параметров, при
    которых значения, возвращаемые функционалом, окажутся в совокупности минимальными.
    """
    params = _ensure_tuple(params)

    def _update_model(values):
        for param, value in zip(params, values):
            setattr(model, param, value)
        model.ensure_physical_ranges(physical_range_policy)

    def _get_distances(values):
        logging.log(LEVEL_PROGRESS, f'Calculating distances at {params} = {values}')
        _update_model(values)
        return functional()

    logging.info(f'Searching for optimal {params}')
    result = leastsq(
        _get_distances,
        tuple(getattr(model, param) for param in params),
        **kwargs,
    )
    _update_model(result[0])


def find_optimal_approximation(model, params, curves, **kwargs):
    """С помощью метода наименьших квадратов подбирает такие значения запрошенных параметров, при
    которых модель наилучшим образом аппроксимирует заданный набор кривых.
    """
    curves = fit_curves_with_model(curves, model)
    values_iterator = _make_values_iterator(curves)
    transform_value = math.log if isinstance(curves, RateCurves) else (lambda pt: pt)

    def _get_model_to_curves_distances():
        result = []
        for key, curve in curves:
            model_values = values_iterator(model, key, curve)
            result.extend(
                transform_value(model_value) - transform_value(orig_pt[1])
                for orig_pt, model_value in zip(curve, model_values)
            )
        return result

    logging.info(f'Approximating {curves.__class__.__name__} with {len(curves)} keys')
    find_optimal_params(model, params, _get_model_to_curves_distances, **kwargs)


def identify_model(
    model,
    main_params,
    *other_params,
    rate_curves=None,
    strain_curves,
    grain_growth_curves=None,
    grain_growth_params=tuple(),
    **kwargs,
):
    """Осуществляет поиск оптимальных значений параметров модели (идентификацию) на основе наборов
    скоростных кривых, кривых деформирования и кривых роста зёрен для одного и того же материала.
    Используется авторский алгоритм, построенный вокруг идеи группировки параметров модели по
    описываемому ими эффекту, и его различные модификации, описанные в оригинальной работе.
    """
    main_params = _ensure_tuple(main_params)
    grain_growth_params = _ensure_tuple(grain_growth_params)
    other_params = [_ensure_tuple(params) for params in other_params]

    if (grain_growth_curves is not None) != bool(grain_growth_params):
        raise ValueError(
            'Arguments "grain_growth_curves" and "grain_growth_params" must be passed together'
        )

    if rate_curves is not None:
        find_optimal_approximation(model, main_params, rate_curves, **kwargs)
        logging.info(f'After rate step: {model}')

    find_optimal_approximation(model, main_params, strain_curves, **kwargs)
    logging.info(f'After strain step: {model}')

    if grain_growth_curves is not None:
        find_optimal_approximation(model, grain_growth_params, grain_growth_curves, *kwargs)
        logging.info(f'After grain growth step: {model}')

    params = list(main_params)
    for num, params_group in enumerate(other_params):
        params.extend(params_group)
        find_optimal_approximation(model, params, strain_curves, **kwargs)
        logging.info(f'After #{num} additional step: {model}')
    return model


def iter_model_yield_stresses(model, initial_grain_size, curve):
    """Определяет по модели значения напряжений течения для скоростей деформации, соответствующих
    заданной скоростной кривой.
    """
    model_curve = build_rate_curve(
        MaxLoadRateCurveBuilder,
        tuple(pt[0] for pt in curve),
        make_specimen_factory(model, initial_grain_size),
    )
    for orig_pt, model_pt in zip(curve, model_curve):
        if orig_pt[0] != model_pt[0]:
            raise RuntimeError(f'Unexpected built point abscissa: {orig_pt} vs. {model_pt}')
        yield model_pt[1]


def iter_model_stresses(*args, **kwargs):
    """Определяет по модели значения напряжений для деформаций, соответствующих заданной кривой
    деформирования.
    """
    return _iter_size_rate_curve_values(build_strain_curve, *args, **kwargs)


def iter_model_grain_sizes(*args, **kwargs):
    """Определяет по модели значения текущего размера зерна для моментов времени, соответствующих
    заданной кривой роста зёрен.
    """
    return _iter_size_rate_curve_values(build_grain_growth_curve, *args, **kwargs)


def _iter_size_rate_curve_values(curve_builder, model, key, curve):
    """Определяет по модели значения абсцисс для ординат, соответствующих заданной кривой."""
    model_curve = curve_builder(
        key.rate,
        max(pt[0] for pt in curve),
        make_specimen_factory(model, key.size),
    )
    return (model_curve.linear_approx(pt[0]) for pt in curve)


def _make_values_iterator(curves):
    """В зависимости от типа кривых возвращает соответствующую функцию, определяющую по модели
    значения ординат точек с абсциссами, соответствующими определённой кривой.
    """
    if isinstance(curves, RateCurves):
        return iter_model_yield_stresses
    elif isinstance(curves, StrainCurves):
        return iter_model_stresses
    elif isinstance(curves, GrainGrowthCurves):
        return iter_model_grain_sizes
    else:
        raise ValueError(f'Unknown curves type {curves.__class__.__name__}')


def _ensure_tuple(params):
    """При необходимости переводит параметры в виде строки с пробелами в список строк."""
    if isinstance(params, str):
        params = params.strip().split()
    return tuple(params)
