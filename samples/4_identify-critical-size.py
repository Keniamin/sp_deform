"""Пример 4: определение критического размера зерна на основе сравнения расчётного распределения
зёрен по размерам (без учёта измельчения) с экспериментальным. Идентификация параметров в уравнении
критического размера с помощью МНК. Расчёт с измельчением по оригинальной модели Ghosh, Raj.
"""

import logging

from scipy.optimize import leastsq

from sp_deform import PiecewiseRateCurveBuilder, build_rate_curve, make_strain_rates
from sp_deform.curves import (
    SortedCurve,
    RateSensitivityCurves, ContinuousGrainsDistributionCurves,
    CurvesPlot,
)
from sp_deform.models import (
    lin_dunne, refinement,
    ElasticStressModel,
    fit_curves_with_model, make_specimen_factory,
)
from sp_deform.refinement import run_refinement_test
from sp_deform.utils import load_data_file

logging.basicConfig(level=logging.INFO)

# Заранее сохраняем параметры для рисования пунктирной линии, чтобы не повторять их каждый раз
DASH_LINE = dict(
    showlegend=False,
    mode='lines',
    line=dict(
        dash='dash',
        color='black',
        width=0.5,
    ),
)

# Обычная модель материала (без учёта измельчения зёрен)
class DunneModel(
    ElasticStressModel,
    lin_dunne.HardeningModel,
    lin_dunne.GrainGrowthModel,
    lin_dunne.SinhDeformationModel,
):
    pass

# Модель материала по работам Dunne и соавторов с добавлением измельчения по модели Ghosh, Raj
# NB: так можно расширять уже "собранные" модели дополнительными функциями из других моделей
class RefinementModel(DunneModel, refinement.GhoshRajRefinementModel):
    pass


material = 'ti6al4v_1173K'
curves = ContinuousGrainsDistributionCurves(material)
# В статье распределение представлено в виде непрерывных кривых. Для перехода к формату групп зёрен
# такое распределение необходимо проинтегрировать. Чем меньше будет шаг интегрирования, тем более
# точно (детально) будет описываться микроструктура, но расчёты будут занимать больше времени
initial_distr = curves.initial.integrate(grid_step=0.25)

models = load_data_file(f'models/{material}.py')
model = DunneModel(**models['normalization'], **models['dunne']['full'])

# Для начала построим расчётные распределения без учёта измельчения и сравним с экспериментальными
for key, curve in fit_curves_with_model(curves, model):
    # Пропускаем начальное распределение
    if key.strain == 0:
        continue
    model_curve = run_refinement_test(model, initial_distr, key.rate, key.strain)

    plot = CurvesPlot(CurvesPlot.Distributions.COUNT)
    plot.add_distribution(initial_distr, color='black')
    plot.add_distribution(curve.integrate(grid_step=0.25), color='green')
    plot.add_distribution(model_curve, color='red')
    plot.show()

# На последнем графике (соответствующем скорости 0.1 с⁻¹) ищем точку пересечения рассчитанного
# распределения с экспериментальным. Эта точка соответствует ожидаемому критическому размеру.
# Дополнительные размеры нужны для создания опорных точек для МНК. Более подробное описание
# алгоритма см. в оригинальной диссертационной работе
critical_size = 7.75
other_sizes = [3, 5, 12, 16]

# NB: нормированные значения! Соответствуют 0.01–1.0 с⁻¹
strain_rates = make_strain_rates(10, 1000, divisor=10 ** (1 / 3))
critical_rate = 100

reference_curve = (
    build_rate_curve(
        PiecewiseRateCurveBuilder,
        strain_rates,
        make_specimen_factory(model, critical_size),
    )
    .make_rate_sensitivity_curve()
)

plot = CurvesPlot(RateSensitivityCurves)
plot.add_sorted_curve(reference_curve, color='orange')

# Собираем опорные точки для МНК: определяем значение параметра скоростной чувствительности m,
# соответствующее известной паре "критический размер зерна — критическая скорость деформации" и
# находим скорости деформации, соответствующие такому же значению m для других размеров зерна
reference_m = reference_curve.linear_approx(critical_rate)
critical_points = [(critical_size, critical_rate)]
for size in other_sizes:
    curve = (
        build_rate_curve(
            PiecewiseRateCurveBuilder,
            strain_rates,
            make_specimen_factory(model, size),
        )
        .make_rate_sensitivity_curve()
    )
    plot.add_sorted_curve(curve, color='darkcyan')
    critical_points.append(
        (size, curve.get_value_position(reference_m, curve.Branch.RIGHT))
    )

# Добавляем на график пунктирные линии, выделяющие известную опорную точку
plot.add_scatter(x=(strain_rates[0], strain_rates[-1]), y=(reference_m, reference_m), **DASH_LINE)
plot.add_scatter(x=(critical_rate, critical_rate), y=(reference_m, 0), **DASH_LINE)
plot.show()

# Обогащаем модель параметрами критического размера по умолчанию
model = RefinementModel(
    **model.to_dict(),
    reference_critical_rate=1,
    mu=0,
)

def get_distances(params):
    model.reference_critical_rate, model.mu = params
    return [rate - model.calc_critical_rate(size) for size, rate in critical_points]

# Ищем с помощью МНК параметры, при которых уравнение критической скорости деформации оптимально
# аппроксимирует найденные опорные точки
result = leastsq(get_distances, (model.reference_critical_rate, model.mu))
model.reference_critical_rate, model.mu = result[0]

# Отображаем на графике опорные точки, кривую с найденными параметрами. Пунктиром подсвечена
# известная точка (определённая по графикам распределения зёрен)
result_curve = SortedCurve(
    (size, model.calc_critical_rate(size))
    for size in range(3, 17)
)

plot = CurvesPlot(None)
plot.update_xaxes(title='d, μm')
plot.update_yaxes(title='𝜀̇, s⁻¹', type='log', exponentformat='power')
plot.add_sorted_curve(SortedCurve(critical_points), color='black', use_only_markers=True)
plot.add_sorted_curve(result_curve)
plot.add_scatter(
    x=(other_sizes[0], critical_size, critical_size),
    y=(critical_rate, critical_rate, strain_rates[0]),
    **DASH_LINE,
)
plot.show()

# Выводим параметры найденной модели
print(model)

# Визуализируем результат расчёта распределения с измельчением зёрен по модели Ghosh, Raj с
# оригинальными параметрами самой модели и найденными выше параметрами критического размера зерна
for key, curve in fit_curves_with_model(curves.filter(rate=0.1), model):
    model_curve = run_refinement_test(model, initial_distr, key.rate, key.strain)

    plot = CurvesPlot(CurvesPlot.Distributions.COUNT)
    plot.update_yaxes(range=(0, 25))
    plot.add_distribution(initial_distr, color='black')
    plot.add_distribution(curve.integrate(grid_step=0.25), color='green')
    plot.add_distribution(model_curve.aggregate(grid_step=0.25), color='blue')
    plot.show()
