"""Пример 2: верификация модели путём сравнения предсказания с моделями, идентифицированными по
различным подмножествам экспериментальных данных.
"""

import logging

from sp_deform.curves import StrainCurves, CurvesPlot
from sp_deform.models import (
    lin_dunne,
    # Некоторые важные (часто применяющиеся) классы и функции можно импортировать из общего модуля
    # Это избавляет от необходимости помнить, в каком именно под-модуле они находятся
    ElasticStressModel, ErrorMetric, MetricsPlot,
    approximate_curves, get_model_metrics,
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
strain_curves = StrainCurves(material)
# Приводящиеся в литературе экспериментальные данные можно оцифровать различными способами
# Например, точки на кривых могут быть распределены равномерно или логарифмически
log_curves = strain_curves.extract_log(divisor=2, min_abscissa=0.01)
uniform_curves = strain_curves.extract_uniform(count=7)

# Чтобы каждый раз не тратить время заново на идентификацию модели, можно использовать заранее
# сохранённые в файлы значения параметров
models = load_data_file(f'models/{material}.py')
model_full = DunneModel(**models['normalization'], **models['dunne']['full_log'])
model_basic = DunneModel(**models['normalization'], **models['dunne']['basic_log'])
model_crit = DunneModel(**models['normalization'], **models['dunne']['critical64_log'])

# Заготовка для графиков (имена и цвета для моделей)
plotting_params = (
    (model_full, 'full', 'blue'),
    (model_basic, 'basic', 'darkcyan'),
    (model_crit, 'crit', 'red'),
)

# Рисуем график приближения экспериментальных данных различными моделями
plot = CurvesPlot(log_curves)
for key, curve in log_curves:
    plot.add_sorted_curve(curve, name=key, use_only_markers=True)
for model, name, color in plotting_params:
    model_curves = approximate_curves(log_curves, model)
    for key, curve in model_curves.normalize():
        plot.add_sorted_curve(curve, name=f'{name}: {key}', color=color)
plot.show()

# Оцениваем и выводим на график погрешности приближения. Каждый тип погрешности требует своего типа
# опорных кривых. Обоснование см. в оригинальной диссертационной работе
for metric_type, reference_curves in (
    (ErrorMetric.AVG, uniform_curves),
    (ErrorMetric.MAX, log_curves),
):
    plot = MetricsPlot(metric_type)
    for model, name, color in plotting_params:
        model_metrics = get_model_metrics(model, metric_type, strain_curves)
        plot.add_metrics(
            model_metrics, name,
            marker_params=dict(symbol='triangle-up', size=8),
            color=color,
        )
    plot.show()
