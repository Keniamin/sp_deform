"""Пример 3: оценка влияния шума в экспериментальных данных на результат идентификации модели."""

import logging
import statistics

from collections import defaultdict

from tabulate import tabulate

from sp_deform.curves import GrainGrowthCurves, RateCurves, StrainCurves, DisturbanceModifier
from sp_deform.models import (
    lin_dunne,
    ElasticStressModel, ErrorMetric,
    identify_model, approximate_curves, get_model_metrics,
)
from sp_deform.utils import load_data_file

logging.basicConfig(level=logging.INFO)

class DunneModel(
    ElasticStressModel,
    lin_dunne.HardeningModel,
    lin_dunne.GrainGrowthModel,
    lin_dunne.SinhDeformationModel,
):
    pass


material = 'ti6al4v_1200K'
original_strain_curves = StrainCurves(material)
rate_curves = RateCurves(material).filter(sizes=(6.4, 9.0, 11.5))
grain_growth_curves_view = GrainGrowthCurves(material).filter(rates=(0, 5e-5, 2e-4, 1e-3))
grain_growth_curves = grain_growth_curves_view.modify(lambda _, curve: curve[1:])

models = load_data_file(f'models/{material}.py')
model_full = DunneModel(**models['normalization'], **models['dunne']['full_log'])
model_strain_curves = approximate_curves(original_strain_curves, model_full).normalize()

# Проиллюстрируем описанную в работе методику: рассмотрим оригинальные экспериментальные данные и
# результат их аппроксимации моделью. На каждый из этих двух наборов кривых будем накладывать шум
# и смотреть на погрешность приближения моделями, идентифицированными по зашумлённым данным
original_errors = defaultdict(lambda: defaultdict(list))
model_errors = defaultdict(lambda: defaultdict(list))
for strain_curves, errors_storage in (
    (original_strain_curves, original_errors),
    (model_strain_curves, model_errors),
):
    uniform_curves = strain_curves.extract_uniform(count=7)
    log_curves = strain_curves.extract_log(divisor=2, min_abscissa=0.01)
    # Поскольку шум, который мы будем накладывать, генерируется случайным образом, погрешность также
    # можно оценивать только статистически. Однако идентификация каждой модели занимает длительное
    # время. Здесь в качестве примера рассчитываются по 3 модели для каждого базового набора, что на
    # компьютере автора занимает чуть больше 2 часов. На практике требуется 10–12 расчётов для того,
    # чтобы результат стабилизировался (см. приложенное изображение; синие точки показывают среднюю
    # погрешность по результатам N расчётов, пунктирные линии — коридор дисперсии). В связи с этим,
    # не стоит удивляться, если результат выполнения данного примера окажется не таким наглядным,
    # как в оригинальной работе. Там для оценки влияния шума использовалось 26 расчётов (для каждого
    # материала и набора кривых), которые для экономии времени запускались параллельно. Реализация
    # параллельного расчёта (и описание ограничений такого подхода) приведена в файле 3-parallel.
    for _ in range(3):
        disturbance_modifier = DisturbanceModifier()
        modified_curves = log_curves.modify(disturbance_modifier)
        model = identify_model(
            DunneModel(
                yield_stress=0.5, reference_rate=1e-3,
                normalized_young_modulus=2000,
                B=0.001,
            ),
            'A B alpha',
            'H_0 H',
            rate_curves=rate_curves,
            strain_curves=modified_curves,
            grain_growth_curves=grain_growth_curves,
            grain_growth_params='D G beta phi',
        )
        for metric_type, reference_curves in (
            (ErrorMetric.AVG, uniform_curves),
            (ErrorMetric.MAX, log_curves),
        ):
            errors = get_model_metrics(model, metric_type, reference_curves)
            for key, error in errors.items():
                errors_storage[key][metric_type].append(error)

def format_stats(lst):
    mean = statistics.mean(lst)
    return f'{mean:.2f}±{statistics.stdev(lst, mean):.2f}'

# Весьма полезный оператор `assert` позволяет добавить в код так называемые sanity check'и или
# "проверки на вменяемость". Это позволяет в автоматическом режиме перепроверять себя и не
# полагаться на "неявные" гарантии, подразумевающиеся по смыслу предыдущего кода (но потенциально
# нарушенные из-за ошибки в логике программы), а явно и недвусмысленно убеждаться в их истинности
assert original_errors.keys() == model_errors.keys()

table = [
    [
        key or 'common',
        format_stats(original_errors[key][ErrorMetric.AVG]),
        format_stats(model_errors[key][ErrorMetric.AVG]),
        format_stats(original_errors[key][ErrorMetric.MAX]),
        format_stats(model_errors[key][ErrorMetric.MAX]),
    ]
    for key in original_errors
]
print(tabulate(table, headers=('\nkey', 'Δavg\norig', '\nmodel', 'Δmax\norig', '\nmodel')))
