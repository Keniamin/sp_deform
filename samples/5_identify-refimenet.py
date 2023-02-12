"""Пример 5: идентификация авторской модели с неполным измельчением групп зёрен, анализ результата."""

import logging

from sp_deform.curves import ContinuousGrainsDistributionCurves, CurvesPlot
from sp_deform.models import ElasticStressModel, lin_dunne, refinement
from sp_deform.models.identification import LEVEL_PROGRESS, find_optimal_params
from sp_deform.refinement import run_refinement_test
from sp_deform.utils import load_data_file

logging.basicConfig(level=LEVEL_PROGRESS)


class DunneModel(
    ElasticStressModel,
    lin_dunne.HardeningModel,
    lin_dunne.GrainGrowthModel,
    lin_dunne.SinhDeformationModel,
):
    pass

class RefinementModel(DunneModel, refinement.GoncharovRefinementModel):
    pass


grid_step = 0.25

material = 'ti6al4v_1173K'
curves = ContinuousGrainsDistributionCurves(material)
initial_distr = curves.initial.integrate(grid_step=grid_step)
target_distr = curves[(0.1, 0.552)].integrate(grid_step=grid_step)

models = load_data_file(f'models/{material}.py')
model = DunneModel(
    **models['normalization'],
    **models['dunne']['full'],
)
no_refinement_distr = run_refinement_test(
    model,
    initial_distr,
    strain_rate=100,
    max_strain=0.552,
).aggregate(grid_step=grid_step)

# NB: при использовании модели с неполным измельчением группы в процессе расчёта активно меняются.
# Если интегрировать начальное и конечное распределения с одинаковым шагом, результат получается
# "шумным" (т.к. при агрегации конечного расчётного распределения некоторые группы могут попасть
# в соседний интервал). Поэтому для повышения точности начальное распределение интегрируется с
# меньшим шагом. При этом, чтобы возможная ошибка при агрегации конечного распределения была
# наименьшей, шаг интегрирования должен быть кратен конечному
smaller_initial_distr = curves.initial.integrate(grid_step=(grid_step / 4))

def get_distribution():
    result = run_refinement_test(
        model,
        smaller_initial_distr,
        strain_rate=100,
        max_strain=0.552,
    )
    return result.aggregate(grid_step=grid_step)

def get_difference():
    model_distr = get_distribution()
    index = 0
    while model_distr[index][0] < target_distr[0][0] - 1e-9:
        # Пропускаем группы в расчётном распределении до начала экспериментального
        index += 1
    difference = []
    for pt in target_distr:
        model_value = 0
        if index < len(model_distr):
            # Для каждой общей группы проверяем, что размер совпадает, и фиксируем разницу
            assert abs(pt[0] - model_distr[index][0]) < 1e-9
            model_value = model_distr[index][1]
        difference.append(pt[1] - model_value)
        index += 1
    return difference

# Инициализируем модель известными параметрами и определяем оптимальные параметры измельчения
model = RefinementModel(
    **models['normalization'],
    **models['dunne']['full'],
    **models['dunne']['critical_size_mixin'],
)
find_optimal_params(model, 's_0 theta r', get_difference)
model_distr = get_distribution()

print(model)
print('Terminal size for model:', model.calc_terminal_size(100))

# Визуализируем конечное расчётное распределение на фоне начального и экспериментального
plot = CurvesPlot(CurvesPlot.Distributions.COUNT)
plot.add_distribution(initial_distr, color='black')
plot.add_distribution(target_distr, color='green')
plot.add_distribution(model_distr, color='blue')
plot.show()

# Визуализируем конечное расчётное распределение на фоне экспериментального и расчётного без учёта
# измельчения зёрен (для оценки эффекта, вносимого учётом измельчения)
plot = CurvesPlot(CurvesPlot.Distributions.COUNT)
plot.add_distribution(target_distr, color='green')
plot.add_distribution(no_refinement_distr, color='red')
plot.add_distribution(model_distr, color='blue')
plot.show()
