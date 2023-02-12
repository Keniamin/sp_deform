"""Пример 3: оценка влияния шума на результат идентификации, версия с параллельным расчётом.
Обратите внимание, что в операционной системе Windows при использовании интерактивного режима
(Jupyter Notebook и его аналоги) данный пример НЕ БУДЕТ РАБОТАТЬ. Более того, вероятно он не просто
сломается ("упадёт" с ошибкой), а наглухо зависнет — что по случайности можно принять за ожидаемое
поведение, поскольку код расчёта действительно работает долго. Будьте внимательны и не попадитесь:
корректно запущенный код практически сразу выводит в лог первые сообщения о запуске расчётов. Если
этих сообщений нет в течение минуты после запуска — вероятнее всего, что-то идёт не так.

Технические детали и объяснение эффекта зависания, можно прочитать здесь:
https://stackoverflow.com/questions/48846085/python-multiprocessing-within-jupyter-notebook
Если вкратце — у системы Windows и семейства *nix принципиально отличается способ порождения новых
процессов. На Windows дочерний процесс не может запустить функцию, которая представлена только в
памяти родительского процесса, — а именно так получается, если просто скопировать код этого примера
в Jupyter. Для корректной работы необходимо, чтобы запускаемая функция была доступна в именованном
файле, лежащем на диске, который дочерний процесс сможет прочитать (импортировать) и выполнить.
"""

import logging
import statistics
import multiprocessing
import os

from collections import defaultdict, namedtuple
from itertools import chain

from tabulate import tabulate

from sp_deform.curves import GrainGrowthCurves, RateCurves, StrainCurves, DisturbanceModifier
from sp_deform.models import (
    lin_dunne,
    ElasticStressModel, ErrorMetric,
    identify_model, approximate_curves, get_model_metrics,
)
from sp_deform.utils import load_data_file

logging.basicConfig(level=logging.INFO)

ErrorsInfo = namedtuple('ErrorsInfo', ('metric_type', 'key', 'error'))


class DunneModel(
    ElasticStressModel,
    lin_dunne.HardeningModel,
    lin_dunne.GrainGrowthModel,
    lin_dunne.SinhDeformationModel,
):
    pass


def get_errors(strain_curves, rate_curves, grain_growth_curves):
    uniform_curves = strain_curves.extract_uniform(count=7)
    log_curves = strain_curves.extract_log(divisor=2, min_abscissa=0.01)
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
    results = []
    for metric_type, reference_curves in (
        (ErrorMetric.AVG, uniform_curves),
        (ErrorMetric.MAX, log_curves),
    ):
        errors = get_model_metrics(model, metric_type, reference_curves)
        for key, error in errors.items():
            results.append(ErrorsInfo(metric_type=metric_type, key=key, error=error))
    return results


def format_stats(lst):
    mean = statistics.mean(lst)
    return f'{mean:.2f}±{statistics.stdev(lst, mean):.2f}'


# В правильно оформленном Python-скрипте "основной" код также должен содержаться внутри функции.
# Это позволяет импортировать данный файл как модуль и использовать другие функции/объекты из него,
# не вызывая автоматического (в момент импорта) срабатывания кода главной функции. Для чего это
# нужно см. в пояснении в начале файла (по поводу исполнения дочерних процессов в системе Windows)
def main():
    material = 'ti6al4v_1200K'
    original_strain_curves = StrainCurves(material)
    rate_curves = RateCurves(material).filter(sizes=(6.4, 9.0, 11.5))
    grain_growth_curves = (
        GrainGrowthCurves(material)
        .filter(rates=(0, 5e-5, 2e-4, 1e-3))
        .modify(lambda _, curve: curve[1:])
    )

    models = load_data_file(f'models/{material}.py')
    model_full = DunneModel(**models['normalization'], **models['dunne']['full_log'])
    model_strain_curves = approximate_curves(original_strain_curves, model_full).normalize()

    test_count = 10

    # NB: по умолчанию вычислительный пул занимает все доступные ядра процессора. При этом другие
    # программы (текстовый редактор, браузер) и даже интерфейс системы могут начать ощутимо
    # "подтормаживать". Если хочется продолжать работу с компьютером во время расчёта и при этом
    # у процессора доступно больше 2 ядер — можно занять расчётами на одно ядро меньше, оставив
    # его для работы системы. При этом время расчёта, очевидно, увеличится, но расчёты все ещё
    # будут идти параллельно (то есть быстрее, чем в исходном примере). Чтобы это сделать нужно
    # добавить при создании пула дополнительный аргумент: `Pool(processes=(os.cpu_count() - 1))`
    pool = multiprocessing.Pool()
    original_results = pool.starmap_async(get_errors, [
        (original_strain_curves, rate_curves, grain_growth_curves)
        for _ in range(test_count)
    ])
    model_results = pool.starmap_async(get_errors, [
        (model_strain_curves, rate_curves, grain_growth_curves)
        for _ in range(test_count)
    ])

    pool.close()  # указывает пулу, что в него больше не будет добавляться задач
    pool.join()  # ожидает завершения исполнения задач, которые уже были поставлены

    original_errors = defaultdict(lambda: defaultdict(list))
    model_errors = defaultdict(lambda: defaultdict(list))
    for results, errors_storage in (
        (original_results.get(), original_errors),
        (model_results.get(), model_errors),
    ):
        for metric_type, key, error in chain.from_iterable(results):
            errors_storage[key][metric_type].append(error)
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


# При импорте данного файла из другого модуля глобальная переменная `__name__` будет содержать имя
# файла. Если же файл будет так называемой "точкой входа" в программу (то есть основным модулем,
# который программа непосредственно запускает), переменная будет содержать специальное значение.
# Именно в этом (и только в этом) случае мы должны запустить здесь исполнение главной функции
if __name__ == '__main__':
    main()
