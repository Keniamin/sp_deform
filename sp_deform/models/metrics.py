# -*- coding: utf8 -*-
"""Функции для вычисления и изображения метрик погрешности приближения модели."""

from enum import Enum

from plotly.graph_objs import Figure

from ..curves import StrainCurves
from ..utils import make_common_layout, update_with_defaults
from .approximation import fit_curves_with_model
from .identification import iter_model_stresses


class ErrorMetric(Enum):
    """Тип метрики погрешности приближения."""
    AVG = 'avg'
    MAX = 'max'


class MetricsPlot(Figure):
    """Класс, реализующий построение графика с метриками аппроксимации."""

    def __init__(self, metric_type):
        """Инициализирует график настройками оси и подписью к ней в соответствии с типом метрики."""
        if metric_type is ErrorMetric.AVG:
            type_text = 'avg'
        elif metric_type is ErrorMetric.MAX:
            type_text = 'max'
        else:
            raise ValueError(f'Unknown error metric type: {metric_type}')

        layout = make_common_layout(add_yaxis=True)
        layout['yaxis']['title']['text'] = f'Δ{type_text}, %'
        super().__init__(layout=layout)

    def add_metrics(
        self,
        model_metrics,
        model_name,
        *,
        color=None,
        use_bars=False,
        line_params=None,
        marker_params=None,
        common_line_params=None,
    ):
        """Добавляет набор метрик на график. По умолчанию набор изображается непрерывной линией.
        Аргументы функции позволяют вместо этого использовать столбчатую диаграмму. Кроме того,
        общую погрешность по набору можно изображать в виде общей линии, пересекающей весь график на
        соответствующем уровне. Также можно настроить произвольные параметры линии/столбцов.
        """
        curve_keys = sorted(
            (curve_key for curve_key in model_metrics.keys() if curve_key is not None),
            key=lambda curve_key: (curve_key.rate, curve_key.size),
        )
        scatter_or_bar = dict(
            x=[str(curve_key) for curve_key in curve_keys],
            y=[model_metrics[curve_key] for curve_key in curve_keys],
            name=model_name,
        )

        if model_metrics.get(None) is not None:
            common_error = model_metrics[None]
            if common_line_params is None:
                scatter_or_bar['x'].insert(0, 'common')
                scatter_or_bar['y'].insert(0, common_error)
            else:
                common_scatter = dict(
                    x=scatter_or_bar['x'],
                    y=[common_error] * len(curve_keys),
                    name=f'{model_name}, common',
                    mode='lines',
                    line=update_with_defaults(
                        common_line_params,
                        color=color,
                        width=0.667,
                    ),
                )
                self.add_scatter(**common_scatter)

        if use_bars:
            scatter_or_bar['marker'] = update_with_defaults(
                marker_params,
                color=color,
                opacity=0.667,
            )
            self.add_bar(**scatter_or_bar)
        else:
            mode = 'lines' if marker_params is None else 'lines+markers'
            scatter_or_bar.update(dict(
                mode=mode,
                line=update_with_defaults(
                    line_params,
                    color=color,
                    width=0.667,
                ),
                marker=update_with_defaults(
                    marker_params,
                    color=color,
                ),
            ))
            self.add_scatter(**scatter_or_bar)

        return self


def get_model_metrics(model, metric_type, reference_curves):
    """Строит для модели заданную метрику погрешности приближения на наборе кривых. Как показано в
    авторской работе, для оценки погрешности по среднему лучше использовать кривые с равномерным
    распределением абсцисс точек, а для погрешности по максимуму — с логарифмическим.
    """
    if not isinstance(reference_curves, StrainCurves):
        raise TypeError(f'Strain curves are required, got {reference_curves.__class__.__name__}')
    metrics = {}
    sum_metric = sum_weight = 0
    for key, curve in fit_curves_with_model(reference_curves, model):
        model_stresses = iter_model_stresses(model, key, curve)
        diffs = (
            abs((approx - orig_pt[1]) / orig_pt[1])
            for orig_pt, approx in zip(curve, model_stresses)
        )
        if metric_type is ErrorMetric.MAX:
            cur_metric, cur_weight = max(diffs), 1
        elif metric_type is ErrorMetric.AVG:
            cur_metric, cur_weight = sum(diffs), len(curve)
        else:
            raise ValueError(f'Unknown error metric type: {metric_type}')
        metrics[key] = 100 * cur_metric / cur_weight
        sum_metric += cur_metric
        sum_weight += cur_weight
    metrics[None] = 100 * sum_metric / sum_weight
    return metrics
