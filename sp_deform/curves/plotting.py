# -*- coding: utf-8 -*-
"""Реализация функций для визуализации моделей."""

from enum import Enum

from plotly.graph_objs import Figure

from ..utils import make_common_layout, update_with_defaults
from .base_classes import SuperplasticityCurves, RateBasedCurves, StrainTimeBasedCurves
from .sets import (
    RateCurves, RateSensitivityCurves, StrainCurves,
    GrainGrowthCurves, ContinuousGrainsDistributionCurves,
)


class CurvesPlot(Figure):
    """Класс, реализующий построение графика с кривыми сверхпластичности."""

    class Distributions(Enum):
        """Указывает, что график изображает распределение количества зёрен либо их объёмной доли по
        размеру.
        """
        COUNT = 'count'
        VOLUME = 'volume'

    def __init__(self, reference):
        """Инициализирует график настройками осей и подписями к ним в соответствии с типом
        изображаемых кривых. Параметр `reference` задаёт тип кривых и может быть одним из объектов:
          * известным набором кривых — будут использованы его тип и параметры;
          * типом набора кривых — будет использован этот тип с параметрами по умолчанию;
          * значением из перечисления `Distributions` — выведет оси соответствующего распределения;
          * `None` — не будет никак подписывать оси, оставляя это на усмотрение вызывающего кода.
        """
        layout = make_common_layout(add_xaxis=True, add_yaxis=True)

        def make_log_axis(axis, title):
            axis['type'] = 'log'
            axis['exponentformat'] = 'power'
            axis['title']['text'] = title

        def make_tozero_axis(axis, title):
            axis['rangemode'] = 'tozero'
            axis['title']['text'] = title

        if isinstance(reference, type) and issubclass(reference, SuperplasticityCurves):
            reference = reference({})  # empty set of given type with default settings

        if isinstance(reference, RateBasedCurves):
            make_log_axis(layout['xaxis'], '𝜀̇, s⁻¹')
        elif isinstance(reference, StrainTimeBasedCurves):
            if reference.base_variable is StrainTimeBasedCurves.BaseVariable.STRAIN:
                make_tozero_axis(layout['xaxis'], '𝜀')
            elif reference.base_variable is StrainTimeBasedCurves.BaseVariable.TIME:
                make_tozero_axis(layout['xaxis'], 't, s')
            else:
                raise ValueError(f'Unknown base variable {reference.base_variable}')
        elif isinstance(reference, (ContinuousGrainsDistributionCurves, CurvesPlot.Distributions)):
            layout['xaxis']['title']['text'] = 'd, μm'

        if isinstance(reference, RateCurves):
            make_log_axis(layout['yaxis'], '𝜎, MPa')
        elif isinstance(reference, RateSensitivityCurves):
            layout['yaxis']['title']['text'] = 'm'
        elif isinstance(reference, StrainCurves):
            make_tozero_axis(layout['yaxis'], '𝜎, MPa')
        elif isinstance(reference, GrainGrowthCurves):
            layout['yaxis']['title']['text'] = 'd, μm'
        elif (
            isinstance(reference, ContinuousGrainsDistributionCurves)
            or reference is CurvesPlot.Distributions.COUNT
        ):
            make_tozero_axis(layout['yaxis'], 'n, pcs')
        elif reference is CurvesPlot.Distributions.VOLUME:
            make_tozero_axis(layout['yaxis'], 'v')

        super().__init__(layout=layout)

    def add_sorted_curve(
        self,
        curve,
        *,
        name=None,
        color=None,
        use_only_markers=False,
        line_params=None,
        marker_params=None,
    ):
        """Добавляет кривую на график. По умолчанию кривая изображается сплошной непрерывной линией.
        Аргументы функции позволяют вместо этого использовать отметки (единичные точки), а также
        задать имя кривой, цвет или настроить произвольные параметры линии/отметок.
        """
        scatter = dict(
            x=curve.x,
            y=curve.y,
            showlegend=(name is not None),
        )
        if name is not None:
            scatter['name'] = str(name)

        if use_only_markers:
            if color is None:
                color = 'gray'
            scatter.update(dict(
                mode='markers',
                marker=update_with_defaults(
                    marker_params,
                    color=color,
                    symbol='x-thin-open',
                    size=8,
                ),
            ))
        else:
            mode = 'lines' if marker_params is None else 'lines+markers'
            scatter.update(dict(
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

        self.add_scatter(**scatter)
        return self

    def add_distribution(
        self,
        curve,
        *,
        color,
        name=None,
        opacity=0.2,
        line_params=None,
    ):
        """Добавляет на график распределение зёрен по размеру, которое изображается в виде заливки
        соответствующей распределению области. Аргументы функции позволяют задать имя распределения,
        цвет и прозрачность заливки и настроить произвольные параметры ограничивающей область линии.
        """
        scatter = dict(
            showlegend=(name is not None),
            mode='lines',
            line=update_with_defaults(
                line_params,
                color=color,
                width=0.667,
            ),
        )
        if name is not None:
            scatter['name'] = str(name)

        if len(curve) < 2:
            scatter.update(dict(
                x=[curve[0][0], curve[0][0]],
                y=[0, curve[0][1]],
            ))
            self.add_scatter(**scatter)
        else:
            sizes = curve.x
            middles = [0.5 * (s1 + s2) for s1, s2 in zip(sizes, sizes[1:])]
            middles.append(2 * sizes[-1] - middles[-1])
            prev_size = 2 * sizes[0] - middles[0]

            scatter_x = scatter['x'] = [prev_size]
            scatter_y = scatter['y'] = [0]
            for ind in range(len(curve)):
                new_size = middles[ind]
                scatter_x.extend([prev_size, new_size])
                scatter_y.extend([curve[ind][1]] * 2)
                prev_size = new_size
            scatter_x.append(prev_size)
            scatter_y.append(0)

            area_scatter = scatter.copy()
            area_scatter.pop('line')
            area_scatter.update(dict(
                showlegend=False,
                hoverinfo='none',
                mode='none',
                fill='toself',
                fillcolor=scatter['line']['color'],
                opacity=opacity,
            ))
            self.add_scatter(**area_scatter)
            self.add_scatter(**scatter)

        return self
