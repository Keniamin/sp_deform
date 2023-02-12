"""Пример 6: сравнение макро-характеристик образца при расчёте по модели без учёта измельчения зёрен
и по модели с его учётом.
"""

import logging

from scipy.stats import norm

from sp_deform import (
    Specimen,
    SpecimenGauge, MicrostructureGauge, combine_gauges,
    load_data_file, get_optimal_time_step,
)
from sp_deform.curves import (
    StrainCurve, GrainGrowthCurve, GrainsDistributionCurve,
    StrainCurves, GrainGrowthCurves,
    CurvesPlot,
)
from sp_deform.models import (
    ElasticStressModel,
    lin_dunne, refinement,
    VolumeFractionWeightedMicrostructureState,
)

logging.basicConfig(level=logging.INFO)


class DunneModel(
    ElasticStressModel,
    lin_dunne.HardeningModel,
    lin_dunne.GrainGrowthModel,
    lin_dunne.SinhDeformationModel,
):
    pass

class RefinementModel(DunneModel, refinement.GoncharovRefinementModel):
    pass


def make_normal_distr(coef, mean=0.0, stddev=1.0, grid_step=0.25):
    # Функция создаёт "купол" нормального распределения с заданными
    # параметрами, проинтегрированного по сетке с указанным шагом
    start = int(norm.ppf(0.001, mean, stddev) // grid_step)
    end = 1 + int(norm.ppf(0.999, mean, stddev) // grid_step)
    result = [((start - 0.5) * grid_step, 0)]
    for multiplier in range(start, end):
        value = (
            norm.cdf((multiplier + 1) * grid_step, mean, stddev)
            - norm.cdf(multiplier * grid_step, mean, stddev)
        )
        result.append(((multiplier + 0.5) * grid_step, coef * value))
    result.append(((end + 0.5) * grid_step, 0))
    return result


# Параметры выбраны для соответствия графику из оригинальной диссертационной работы
initial_distr = GrainsDistributionCurve(
    make_normal_distr(19, 6, 1.25)
    + make_normal_distr(1, 20, 1.25)
)

# Для информации рассчитываем и выводим количественную и объёмную доли зёрен малого размера
aggregated_count = [pt[1] for pt in initial_distr.aggregate(grid_step=15)]
assert len(aggregated_count) == 2
print('Small grains count fraction:', aggregated_count[0] / sum(aggregated_count))

aggregated_volumes = [pt[1] for pt in initial_distr.get_volumes().aggregate(grid_step=15)]
assert len(aggregated_volumes) == 2
print('Small grains volume fraction:', aggregated_volumes[0] / sum(aggregated_volumes))

# Визуализируем само распределение
plot = CurvesPlot(CurvesPlot.Distributions.VOLUME)
plot.add_distribution(initial_distr.get_volumes(), color='black')
plot.show()

def make_test(model):
    strain_rate = 100  # NB: нормированная величина!
    initial_state = VolumeFractionWeightedMicrostructureState(initial_distr.get_volumes())
    # Здесь приводится пример "сырого" моделирования эксперимента (без использования функций-обёрток
    # типа `run_refinement_test`). Для этого создаётся испытуемый образец и датчики, которые будут
    # "считывать" его параметры в ходе моделирования. Далее запускается тест по заданной программе
    # до нужного состояния (например, значения деформации) с применением созданных датчиков. После
    # завершения моделирования на основе показаний датчиков строятся графики
    specimen = Specimen(model, initial_state)
    time_gauge = SpecimenGauge('time')
    strain_gauge = SpecimenGauge('strain')
    stress_gauge = MicrostructureGauge('stress')
    size_gauge = MicrostructureGauge('grain_size')
    specimen.run_test(
        lambda time: strain_rate * time,  # программа испытания
        lambda: specimen.strain >= 1,  # условие завершения
        get_optimal_time_step(strain_rate),  # периодичность проверки условия и заполнения датчиков
        gauges=(time_gauge, strain_gauge, stress_gauge, size_gauge),
    )
    time_scale = 1 / model.reference_rate
    return (
        # При сыром расчёте важно не забывать приводить считанные показания к размерным величинам
        StrainCurve(combine_gauges(strain_gauge, stress_gauge, data_scale=(1, model.yield_stress))),
        GrainGrowthCurve(combine_gauges(time_gauge, size_gauge, data_scale=(time_scale, 1))),
    )

models = load_data_file(f'models/ti6al4v_1173K.py')
noref_strain_curve, noref_grain_growth_curve = make_test(DunneModel(
    **models['normalization'],
    **models['dunne']['full'],
))
ref_strain_curve, ref_grain_growth_curve = make_test(RefinementModel(
    **models['normalization'],
    **models['dunne']['full'],
    **models['dunne']['critical_size_mixin'],
    **models['dunne']['refinement_mixin_strong'],
))

# Визуализируем полученные кривые на общих графиках
plot = CurvesPlot(StrainCurves)
plot.add_sorted_curve(noref_strain_curve, color='red')
plot.add_sorted_curve(ref_strain_curve, color='blue')
plot.show()

plot = CurvesPlot(GrainGrowthCurves)
plot.add_sorted_curve(noref_grain_growth_curve, color='red')
plot.add_sorted_curve(ref_grain_growth_curve, color='blue')
plot.show()
