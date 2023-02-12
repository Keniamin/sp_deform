# -*- coding: utf-8 -*-
"""Реализация базовых классов кривых (диаграмм деформирования) сверхпластичности."""

import math

from enum import Enum


class SortedCurve:
    """Отсортированный набор точек."""
    __slots__ = ('_points',)

    def __init__(self, points):
        """Сортирует переданные точки по возрастанию абсциссы и проверяет, что кривая не пустая."""
        if isinstance(points, SortedCurve):
            fixed_points = points._points
        else:
            fixed_points = tuple(sorted(tuple(item) for item in points))
            if not fixed_points:
                raise ValueError(f'{self.__class__.__name__} can not be empty')
        self._points = fixed_points

    def __len__(self):
        return len(self._points)

    def __iter__(self):
        return iter(self._points)

    def __reversed__(self):
        return reversed(self._points)

    def __getitem__(self, key):
        return self._points[key]

    def __repr__(self):
        return f'<{self.__class__.__name__} with {len(self._points)} points>'

    @property
    def x(self):
        return tuple(pt[0] for pt in self)

    @property
    def y(self):
        return tuple(pt[1] for pt in self)

    def index(self, *args, **kwargs):
        return self._points.index(*args, **kwargs)

    def aggregate(self, *, grid_step):
        """Агрегирует (суммирует) значения точек кривой по сетке с указанным шагом (границы
        интервалов — кратные шагу точки). Диапазон от первой до последней точки — минимально
        возможный, чтобы целиком покрыть кривую с учётом шага сетки.
        """
        multiplier = self[0][0] // grid_step
        result, index = [], 0
        while index < len(self):
            total = 0
            next_bound = (multiplier + 1) * grid_step
            while index < len(self) and self[index][0] < next_bound:
                total += self[index][1]
                index += 1
            result.append(((multiplier + 0.5) * grid_step, total))
            multiplier += 1
        return SortedCurve(result)


class LinearlyApproximatedCurve(SortedCurve):
    """Набор точек, аппроксимирующий некоторую непрерывную зависимость между величинами."""
    integration_result_class = SortedCurve

    def linear_approx(self, base):
        """Получает значение зависимой переменной по базисной, линейно аппроксимируя кривую между
        опорными (известными) точками. Для быстрого нахождения ближайших к искомому значению точек,
        между которыми кривая затем аппроксимируется, используется алгоритм бинарного поиска.
        """
        left, right = 0, len(self) - 1
        if base < self[left][0] or base > self[right][0]:
            raise ValueError(f'Base value {base} is outside of the curve range')
        while right - left > 1:
            mid = (left + right) // 2
            if self[mid][0] < base:
                left = mid
            else:
                right = mid
        if left == right:
            return self[left][1]
        coef = (base - self[left][0]) / (self[right][0] - self[left][0])
        return coef * self[right][1] + (1 - coef) * self[left][1]

    def integrate(self, *, count=None, grid_step=None):
        """Интегрирует кривую методом трапеций с заданным количеством интервалов либо в привязке к
        сетке с указанным шагом (границы интервалов — кратные шагу точки в пределах кривой).
        Возвращает список пар: середина интервала и значение интеграла кривой на нём.
        """
        if (count is not None) == (grid_step is not None):
            raise ValueError('Exactly one of interval count and grid step must be specified')
        if count is not None:
            if count < 1:
                raise ValueError('Interval count must be at least 1 if specified')
            step = (self[-1][0] - self[0][0]) / count
            bounds = [self[0][0] + num * step for num in range(count)]
            bounds.append(self[-1][0])
        if grid_step is not None:
            bounds = []
            # Moving from end to beginning to avoid complicated code dealing with rounding errors
            multiplier = self[-1][0] // grid_step
            while grid_step * multiplier >= self[0][0]:
                bounds.append(grid_step * multiplier)
                multiplier -= 1
            if len(bounds) < 2:
                raise ValueError('Grid step is too big, not any interval lays in the curve range')
            bounds = list(reversed(bounds))

        result, prev_pt, next_index = [], self[0], 1

        def advance(next_pt):
            nonlocal prev_pt
            integral = 0.5 * (next_pt[0] - prev_pt[0]) * (next_pt[1] + prev_pt[1])
            prev_pt = next_pt
            return integral

        for bound_index in range(len(bounds)):
            cur_integral = 0
            while next_index + 1 < len(self) and self[next_index][0] < bounds[bound_index]:
                cur_integral += advance(self[next_index])
                next_index += 1
            coef = (bounds[bound_index] - prev_pt[0]) / (self[next_index][0] - prev_pt[0])
            bound_pt = (bounds[bound_index], coef * self[next_index][1] + (1 - coef) * prev_pt[1])
            cur_integral += advance(bound_pt)
            if bound_index > 0:
                result.append((0.5 * (bounds[bound_index - 1] + bounds[bound_index]), cur_integral))
        return self.integration_result_class(result)


class RateSensitivityCurve(LinearlyApproximatedCurve):
    """Кривая, описывающая скоростную чувствительность материала."""

    class Branch(Enum):
        """Для сверхпластического материала кривая скоростной чувствительности имеет вид "купола" с
        точкой максимума и двумя убывающими ветвями по сторонам от неё. Данное перечисление задаёт
        одну из этих ветвей.
        """
        LEFT = 'left'
        RIGHT = 'right'

    def __init__(self, points):
        """Проверяет, что все значения кривой — положительные числа. Для оптимизации дальнейших
        вычислений ищет и запоминает точку максимума и её индекс, а также точку минимума на каждой
        из ветвей. Стоит отметить, что у правильной кривой точки минимума всегда находятся по краям.
        Однако из-за вычислительных ошибок в области "хвоста" может наблюдаться некоторое "дрожание"
        значений. Поэтому реализация класса нигде не использует факт монотонности значений явно и
        всегда работает с точками кривой в "сыром" виде.
        """
        super().__init__(points)
        for pt in self:
            if pt[1] <= 0:
                raise ValueError(f'{self.__class__.__name__} values must be positive')
        self.max_value_pt = max(self, key=lambda pt: pt[1])
        self.max_value_index = self.index(self.max_value_pt)
        self.left_min_value_pt = min(self[:self.max_value_index + 1], key=lambda pt: pt[1])
        self.right_min_value_pt = min(self[self.max_value_index:], key=lambda pt: pt[1])

    def get_value_position(self, value, branch, strict=True):
        """Находит положение заданного значения на указанной ветви кривой. Уточнение положения между
        ограничивающими значение точками кривой выполняется с помощью линейного приближения. Целевое
        значение должно лежать в пределах от минимального на ветви до максимального на кривой.
        Параметр `strict` позволяет задать поведение в случае нарушения этого требования: при
        `strict=True` (по умолчанию) будет брошено исключение, при `strict=False` будет возвращено
        положение ближайшей по значению точки (то есть, нарушенного минимума/максимума).
        """
        if branch is RateSensitivityCurve.Branch.LEFT:
            min_value_pt = self.left_min_value_pt
            direction = -1
        elif branch is RateSensitivityCurve.Branch.RIGHT:
            min_value_pt = self.right_min_value_pt
            direction = +1
        else:
            raise ValueError(f'Unknown branch {branch}')
        if value >= self.max_value_pt[1]:
            if strict and value > self.max_value_pt[1]:
                raise ValueError(
                    f'Requested value {value} is greater than curve max {self.max_value_pt[1]}'
                )
            return self.max_value_pt[0]
        if value <= min_value_pt[1]:
            if strict and value < min_value_pt[1]:
                raise ValueError(
                    f'Requested value {value} is less than {branch.value} branch min {min_value_pt[1]}'
                )
            return min_value_pt[0]
        cur_index = self.max_value_index
        while self[cur_index][1] > value and 0 <= cur_index + direction < len(self):
            cur_index += direction
        prev_index = cur_index - direction
        coef = (value - self[prev_index][1]) / (self[cur_index][1] - self[prev_index][1])
        return math.exp(
            coef * math.log(self[cur_index][0])
            + (1 - coef) * math.log(self[prev_index][0])
        )


class RateCurve(LinearlyApproximatedCurve):
    """Кривая, описывающая зависимость напряжения течения материала от скорости деформации."""

    def make_rate_sensitivity_curve(self, _raw=False):
        """Дифференцирует кривую в логарифмических координатах, получая соответствующую ей кривую
        скоростной чувствительности материала.
        """
        if len(self) < 2:
            raise ValueError('Can not calculate the derivative of a single point')

        def make_log(pt):
            return tuple(math.log(item) for item in pt)

        result = []
        prev_pt = make_log(self[0])
        for pt in self[1:]:
            cur_pt = make_log(pt)
            result.append((
                math.exp(0.5 * (cur_pt[0] + prev_pt[0])),
                (cur_pt[1] - prev_pt[1]) / (cur_pt[0] - prev_pt[0]),
            ))
            prev_pt = cur_pt

        if _raw:
            return result
        return RateSensitivityCurve(result)


class StrainCurve(LinearlyApproximatedCurve):
    """Кривая, описывающая зависимость напряжения в материале от деформации или времени."""

    def extract_exact(self, abscissas, *, accuracy=None, _raw=False):
        """Извлекает из кривой точки по заданным абсциссам. По умолчанию для получения значений
        ординат будет построено линейное приближение между ближайшими опорными (известными) точками.
        При задании параметра `accuracy` значения будут извлекаться только из опорных точек кривой,
        при этом расстояние от каждой заданной абсциссы до абсциссы ближайшей опорной точки (из
        которой будет взято значение ординаты) не должно превышать значения `accuracy`.
        """
        if accuracy is None:
            result = [
                (abscissa, self.linear_approx(abscissa))
                for abscissa in abscissas
            ]
        else:
            result = []
            for abscissa in abscissas:
                matched_pt = min(self, key=lambda pt: abs(pt[0] - abscissa))
                if abs(matched_pt[0] - abscissa) > accuracy:
                    raise ValueError(
                        f'Requested abscissa value {abscissa} does not'
                        f' match any point with accuracy {accuracy};'
                        f' nearest point is {matched_pt}'
                    )
                result.append((abscissa, matched_pt[1]))
        if _raw:
            return result
        return StrainCurve(result)

    def extract_uniform(self, count, **kwargs):
        """Извлекает из кривой указанное количество точек с абсциссами, равномерно распределёнными
        от нуля до максимальной на кривой. См. описание функции `extract_exact` для информации о
        дополнительных параметрах.
        """
        max_abscissa = self[-1][0]
        abscissas = [max_abscissa * ind / count for ind in range(1, count)]
        abscissas.append(max_abscissa)
        return self.extract_exact(abscissas, **kwargs)

    def extract_log(self, *, divisor, count=None, min_abscissa=None, **kwargs):
        """Извлекает из кривой точки с абсциссами, логарифмически убывающими с указанным шагом от
        максимальной на кривой до заданной минимальной либо до нахождения заданного количества.
        См. описание функции `extract_exact` для информации о дополнительных параметрах.
        """
        if (min_abscissa is None) == (count is None):
            raise ValueError(f'Exactly one of "count" and "min_abscissa" arguments must be passed')
        abscissas = []
        abscissa = self[-1][0]
        while (
            (min_abscissa is None or abscissa >= min_abscissa)
            and (count is None or len(abscissas) < count)
        ):
            abscissas.append(abscissa)
            abscissa /= divisor
        return self.extract_exact(abscissas, **kwargs)


class GrainGrowthCurve(LinearlyApproximatedCurve):
    """Кривая, описывающая зависимость размера зерна в материале от деформации или времени."""


class GrainsDistributionCurve(SortedCurve):
    """Кривая, описывающая распределение количества зёрен в зависимости от их размера."""

    def __init__(self, points, *, cubes_sum=None):
        """Вычисляет и запоминает полный объём (с точностью до постоянного коэффициента)."""
        super().__init__(points)
        if cubes_sum is not None:
            self.cubes_sum = cubes_sum
        elif isinstance(points, GrainsDistributionCurve):
            self.cubes_sum = points.cubes_sum
        else:
            self.cubes_sum = sum((size ** 3) * count for size, count in self)

    def get_volumes(self):
        """"Возвращает распределение, пересчитанное в виде доли, занимаемой зёрнами заданного
        размера от общего объёма образца.
        """
        return SortedCurve(
            (size, (size ** 3) * count / self.cubes_sum)
            for size, count in self
        )


class ContinuousGrainsDistributionCurve(LinearlyApproximatedCurve):
    """Кривая, описывающая распределение зёрен по размерам в виде непрерывной зависимости."""
    integration_result_class = GrainsDistributionCurve
