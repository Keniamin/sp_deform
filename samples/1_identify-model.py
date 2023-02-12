"""Пример 1: идентификация модели на основе кривых деформирования материала с помощью авторского
алгоритма. Построение графиков приближения экспериментальных данных результирующей моделью.
Сравнение моделей между собой.
"""

import logging

from sp_deform.curves import RateCurves, StrainCurves, GrainGrowthCurves, CurvesPlot
from sp_deform.models import lin_dunne, ElasticStressModel
from sp_deform.models.approximation import approximate_curves
from sp_deform.models.metrics import ErrorMetric, get_model_metrics
from sp_deform.models.identification import LEVEL_PROGRESS, identify_model

# Можно заменить на `level=logging.INFO` для вывода только результатов шагов (без текущего прогресса)
# или `logging.DEBUG` для вывода полной отладочной информации обо всех моделирующихся испытаниях
logging.basicConfig(level=LEVEL_PROGRESS)

# Собираем класс модели из базовых классов, описывающих различные аспекты поведения материала
class DunneModel(
    ElasticStressModel,
    lin_dunne.HardeningModel,
    lin_dunne.GrainGrowthModel,
    lin_dunne.SinhDeformationModel,
):
    pass


# Загружаем экспериментальные данные по деформации титана ВТ6 при температуре 927 градусов цельсия
material = 'ti6al4v_1200K'
strain_curves = StrainCurves(material).extract_log(
    # Исходные кривые данного набора изображены в непрерывном виде. При оцифровке на них взяты и
    # логарифмически, и равномерно распределённые точки (см. оригинальную диссертационную работу).
    # Вызов данной функции оставляет логарифмически распределённые, а параметр `accuracy` говорит о
    # том, что нужно брать только оригинальные точки из набора (не пользуясь линейным приближением)
    divisor=2,
    min_abscissa=0.01,
    accuracy=1e-3,
)
rate_curves = RateCurves(material).filter(sizes=(6.4, 9.0, 11.5))
grain_growth_curves_view = GrainGrowthCurves(material).filter(rates=(0, 5e-5, 2e-4, 1e-3))
# Первая точка на кривых роста зёрен соответствует начальному состоянию. Она неизменна и не зависит
# от параметров модели, а потому не должна участвовать в построении приближения
grain_growth_curves = grain_growth_curves_view.modify(lambda _, curve: curve[1:])

# Создаём "базовую" модель — начальное приближение
model = DunneModel(
    # Нормирующие значения и модуль Юнга (константа материала)
    yield_stress=0.5, reference_rate=1e-3,
    normalized_young_modulus=2000,
    # Параметры A и B являются мультипликативными коэффициентами в общем выражении, поэтому если
    # начать с "чистого" универсального приближения алгоритм не способен будет сделать ни одного
    # шага, поскольку никакое изменение A или B по отдельности не сдвинет кривую с нулевой точки
    B=0.001,
)

# Запускаем непосредственно алгоритм идентификации
identify_model(
    model,
    'A B alpha',
    'H_0 H',
    rate_curves=rate_curves,
    strain_curves=strain_curves,
    grain_growth_curves=grain_growth_curves,
    grain_growth_params='D G beta phi',
)

# Полученная модель отличается от приведённой в оригинальной диссертационной работе. Это связано с
# тем, что при подготовке к публикации кода автор повторно (и более точно) оцифровал используемые
# экспериментальные данные. Тем не менее, результат моделирования практически эквивалентен (ошибка
# составляет порядка 1%), в чём можно убедиться как "на глаз" сравнив графики, так и формально с
# помощью метрик погрешности аппроксимации

# Значения параметров модели из оригинальной работы
dissertation_model = DunneModel(
    yield_stress=0.5, reference_rate=1e-3,
    normalized_young_modulus=2000,
    A=67.6, B=0.0532, alpha=2.35,
    D=0.367, G=10.3, beta=2.09e-5, phi=0.676,
    H_0=4.89, H=3.56,
)

# Нарисуем результат приближения напряжений и роста зёрен обеими моделями
for plot_curves in (strain_curves, grain_growth_curves_view):
    # Подготавливаем графики: добавляем экспериментальные кривые
    plot = CurvesPlot(plot_curves)
    for key, curve in plot_curves:
        plot.add_sorted_curve(curve, name=key, use_only_markers=True)
    # Добавляем результаты приближения моделями
    for plot_model, name, color in (
        (model, 'result', 'blue'),
        (dissertation_model, 'dissertation', 'red'),
    ):
        for key, curve in approximate_curves(plot_curves, plot_model).normalize():
            plot.add_sorted_curve(curve, name=f'{name}: {key}', color=color)
    # При использовании интерактивной среды разработки (Jupyter и подобные) функция `show` отобразит
    # график непосредственно в выходном блоке "ячейки" кода, с возможностью взаимодействия: показ
    # значений точек при наведении курсора, включение/выключение отдельных линий, выделение области
    # на графике для детального рассмотрения. При использовании из консоли вместо этого лучше
    # сохранять график в файл для последующего изучения. Для этого подойдёт функция `write_html`,
    # которая экспортирует график как интерактивную html-страницу (возможности аналогичны описанным
    # выше, для просмотра нужно использовать браузер), либо `write_image`, которая сохранит график в
    # статичное изображение, пригодное для просмотра и правки в любом графическом редакторе, вставки
    # в Word/TeX для статьи и так далее. Подробное описание и возможные параметры указанных функций
    # можно узнать из документации: https://plotly.com/python-api-reference/plotly.io.html
    plot.show()

# Посчитаем разницу между моделями: оценим погрешность приближения новой моделью кривых, построенных
# по оригинальной модели на основе экспериментальных данных
diff_metrics = get_model_metrics(
    model,
    ErrorMetric.AVG,
    approximate_curves(strain_curves, dissertation_model).extract_uniform(count=7),
)
for key in diff_metrics:
    print(
        key or 'common\t\t',
        diff_metrics[key],
        sep='\t',
    )
