## Общие сведения

Данный программный модуль основан на [диссертационной работе](https://istina.msu.ru/dissertations/389216283/) И.А. Гончарова "Моделирование влияния микроструктурных механизмов на поведение материалов при сверхпластическом деформировании" (МГУ, 2021).

Модуль разработан на языке Python. Это интерпретируемый свободно распространяемый язык программирования, широко использующийся для научных и инженерных расчётов. Существует большое количество сторонних модулей (библиотек) для Python, реализующих различный функционал, в том числе математические вычисления и численные методы, например, решение дифференциальных уравнений.

Основными компонентами представленного модуля являются:
- моделирование сверхпластического деформирования с учётом эволюции микроструктуры;
- реализация авторского алгоритма идентификации параметров моделей сверхпластичности.

Вспомогательными компонентами являются:
- оцифрованные кривые сверхпластичности из ряда работ по сверхпластичности (см. ссылки в соответствующих файлах);
- реализация классов, программно описывающих кривые сверхпластичности и наборы таких кривых, а также инструменты их визуализации;
- функционал наложения "шума" на кривые сверхпластичности для оценки его влияния на результат идентификации;
- классы и функции для анализа моделей, в том числе реализация предложенных в работе метрик погрешности аппроксимации.

По любым вопросам, связанным с использованием данного модуля, можно обратиться к автору по электронной почте: <rtif91@gmail.com>

## Лицензия

Данное ПО поставляется в формате "as is", без каких-либо гарантий, явно выраженных или подразумеваемых. Программный модуль является свободным для некоммерческого использования, включая скачивание, запуск и модификацию, а также создание и распространение производных продуктов на его основе. Коммерческое применение возможно при условии получения письменного согласия автора.

При публикации результатов, полученных с использованием данного ПО, необходимо добавлять ссылки на оригинальную диссертационную работу и на корневой репозиторий проекта: <https://github.com/Keniamin/sp_deform>

## Подготовка к работе

Простейший способ получить код модуля — скачать его целиком в виде архива с помощью кнопки `Code` > `Download ZIP` на [главной странице](https://github.com/Keniamin/sp_deform) репозитория. Кроме того, можно получить код модуля для разработки через систему контроля версий по [инструкции](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories): с помощью консольной утилиты git либо собственного [клиента GitHub](https://docs.github.com/en/github-cli/github-cli/about-github-cli).

Для работы Python необходимо установить интерпретатор.
- В системе Windows версии 8+ интерпретатор можно бесплатно установить из встроенного магазина приложений Microsoft Store.
- В системах семейства *nix (в том числе MacOS) он часто включается в стандартную поставку, однако может оказаться достаточно старой версии. Более свежая версия может быть в системном репозитории пакетов. За подробностями обратитесь к справочным материалам конкретной системы и используемого в ней пакетного менеджера.
- Наконец, различные версии интерпретатора можно скачать с [официального сайта](https://www.python.org/downloads/) и установить самостоятельно по инструкции. Программный модуль разработан и протестирован на версии Python 3.10.0, но, вероятно, заработает и на некоторых более старых версиях Python 3.

При наличии интерпретатора можно запускать Python-скрипты и работать с модулями напрямую из консоли. Однако на практике оказывается более удобно воспользоваться подходящим инструментарием:
- интегрированной средой разработки (IDE) с поддержкой Python, например, [Spyder](https://www.spyder-ide.org/);
- интерактивным интерпретатором Jupyter, который можно [использовать](https://jupyter.org/try) в "облаке" прямо из браузера, без установки на компьютер (однако доступные вычислительные мощности в таком случае сильно ограничены), либо [скачать](https://jupyter.org/install) в виде Python-модуля и запустить на собственном компьютере как сервер (открывается так же в браузере, но использует вычислительные ресурсы и файловую систему компьютера);
- IDE со встроенной поддержкой Jupyter, например, плагин для уже упомянутого Spyder или редактор [Visual Studio Code](https://code.visualstudio.com/) с соответствующим расширением, которое позволяет редактировать и запускать Jupyter Notebook'и с подсветкой синтаксиса, навигацией по коду, авто-дополнением имён, всплывающими подсказками из документации и встроенной проверкой некоторых базовых ошибок (этот вариант использовал автор модуля при разработке);
- программным "комбайном", таким как [Anaconda](https://www.anaconda.com/), предоставляющая полновесное рабочее окружение по принципу "ready to use" и содержащая сразу и базовый интерпретатор, и настроенную среду разработки, и большое количество разнообразных модулей-библиотек, скачиваемых из собственного репозитория проекта.

После выбора и установки подходящего инструмента остаётся установить код модуля, сделав его "видимым" для системы. Также в этот момент будут автоматически скачаны и установлены зависимости, то есть необходимые для работы модуля вспомогательные библиотеки. Установка модулей в экосистеме Python обычно осуществляется с помощью пакетного менеджера [pip](https://pip.pypa.io/en/stable/), который можно запустить напрямую из консоли. Либо, при работе через Jupyter, можно воспользоваться так называемой ["магической командой"](https://ipython.readthedocs.io/en/stable/interactive/magics.html), для чего необходимо в отдельной ячейке написать:
```
%pip install <path/to/module>
```

Если планируется модифицировать код модуля, рекомендуется использовать дополнительный флаг `install --editable`. В таком случае модуль будет установлен в "режиме разработчика": при изменении файлов пакета они будут сразу доступны системе, без необходимости каждый раз переустанавливать модуль. Скрипты в таком случае "подхватят" новые версии файлов автоматически при следующем запуске. При работе через Jupyter достаточно будет перезапустить ядро или воспользоваться функцией `reload()` из встроенного модуля `importlib`, чтобы обновить модуль прямо в памяти процесса.

Обратите внимание, что при использовании общего компьютера несколькими пользователями иногда можно столкнуться с конфликтом зависимостей, когда используемые разными пользователями модули потребуют разных версий зависимых библиотек. Например, некоторый давно разработанный модуль не работает со свежей версией библиотеки, а новый модуль, наоборот, использует какие-то возможности новой версии той же библиотеки и не может работать со старой. Поэтому на таких компьютерах библиотеки лучше устанавливать в режиме "только для текущего пользователя". В частности, менеджер `pip` автоматически использует такой режим при обычном запуске от имени пользователя (то есть, без использования режима администратора в Windows или команды `sudo` в *nix). А вот установка зависимостей с помощью системного менеджера пакетов в *nix чаще всего устанавливает пакет конкретной версии для всех пользователей системы сразу.

В том случае, когда даже одному пользователю необходимо работать с несколькими Python-проектами с разными зависимостями, можно воспользоваться так называемыми виртуальными окружениями, обеспечивающими полную изоляцию библиотек между проектами. В базовом варианте их поддержка [доступна](https://docs.python.org/3/tutorial/venv.html) в Python "из коробки", более сложная реализация с дополнительными удобствами существует в виде [отдельного модуля](https://virtualenv.pypa.io/). Виртуальные окружения исходно предназначены для работы в консоли, однако Jupyter и некоторые IDE также могут быть настроены для работы с ними. Такая настройка существенно зависит от способа установки и запуска интерпретатора Python, поэтому за подробностями лучше обратится к документации модуля (по ссылкам выше) и конкретной используемой IDE.

## Структура репозитория

Общая структура проекта выглядит следующим образом:

- `curves` — оцифрованные кривые сверхпластичности из литературы;
- `models` — параметры моделей разных материалов, идентифицированные с помощью авторского алгоритма;
- `samples` — примеры работы модуля с сопроводительными комментариями;
- `sp_deform` — основной код модуля:
    - корневая папка содержит реализацию базовых функций и классов для моделирования сверхпластичности;
    - папка `curves` содержит код классов, описывающих кривые сверхпластичности и наборы таких кривых, инструменты для работы с ними и их визуализации;
    - папка `models` содержит реализацию моделей сверхпластического деформирования (авторских и известных из литературы), а также инструменты анализа моделей и идентификации входящих в них параметров.

Дополнительно каждый модуль (файл), класс и функция в коде сопровождаются так называемой docstring (строкой документации), описывающей содержимое или предназначение данного объекта.
