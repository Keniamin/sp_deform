# -*- coding: utf-8 -*-
"""Реализация класса образца в процессе его испытания."""

import logging
import math
import time


class Specimen(object):
    """Класс, описывающий деформируемый образец."""

    def __init__(self, model, microstructure_state, initial_length=1):
        """Создаёт образец с заданной моделью материала и начальным состоянием микроструктуры.
        Опционально задаётся начальная длина.
        """
        self.time = 0.0
        self.model = model
        self.microstructure_state = microstructure_state
        self._initial_length = initial_length
        self._test_program = None

    @property
    def strain(self):
        """Вычисляет текущую деформацию образца."""
        return self.get_strain(self.time)

    @property
    def length(self):
        """Вычисляет текущую длину образца."""
        return self.get_length(self.time)

    def get_strain(self, time):
        """Вычисляет деформацию образца в заданный момент времени в соответствии с программой
        деформирования.
        """
        if self._test_program is None:
            raise RuntimeError('Specimen test was not started yet')
        return self._test_program(time)

    def get_length(self, time):
        """Вычисляет длину образца в заданный момент времени в соответствии с программой
        деформирования.
        """
        return self._initial_length * math.exp(self.get_strain(time))

    def run_test(self, test_program, end_condition, time_step, gauges):
        """Запускает испытание образца по заданной программе деформирования, которое продолжается
        пока не будет выполнено условие окончания, заданное в виде произвольной функции от образца.
        Условие проверяется в процессе испытания с шагом по времени `time_step`. Также с этим шагом
        с образца снимаются показания переданных датчиков.
        """
        start_time = time.time()
        self._test_program = test_program
        self._fill_gauges(gauges, only_empty=True)
        logging.debug(
            f'Specimen 0x{id(self):X} test program starts'
            f' at time {self.time:.3} with strain {self.strain:.3}'
        )
        while not end_condition():
            self.microstructure_state.make_step(self, time_step)
            self.time += time_step
            self._fill_gauges(gauges)
        test_duration = time.time() - start_time
        logging.debug(
            f'Specimen 0x{id(self):X} tested in {test_duration:.3f} seconds'
            f' up to time {self.time:.3} with strain {self.strain:.3}'
        )

    def _fill_gauges(self, gauges, only_empty=False):
        """Записывает текущее состояние образца с точки зрения заданных датчиков. Параметр
        `only_empty` нужен для корректной инициализации значений в случае, когда один и тот же
        датчик используется в нескольких последовательных испытаниях образца.
        """
        for gauge in gauges:
            if only_empty and not gauge.empty:
                continue
            gauge.store_reading(self)
