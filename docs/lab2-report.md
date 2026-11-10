# Лабораторная №2 — отчёт по этапам

Этап 1: 23.10.2026; MCP и агент чтения: 24.10.2026. Ниже сохранён отчёт первого
этапа; актуальные результаты этапов 2–3 приведены в конце. Полная Lab 2 ещё не завершена.
Задание: [методичка](labs-guide.md#lab2);
контракты: [spec.md](../spec.md), этапы: [plan.md](../plan.md).

## Исходное состояние

Текущая ветка `lab-2`, HEAD `df07a8d` (Initial commit). Она содержала эхо-шаблон,
а не готовую Lab 1. Пользователь уже подготовил в индексе `docs/labs-guide.md` и
`docs/evaluation/lab2-routing.json`; эти изменения сохранены без редактирования.
Локальная `lab-1` на `bbe38e5` содержала режимы study/translate/plan, историю,
настройки temperature, команды reset/settings, LLM-клиент и миграцию 001.

До изменений: `python -m pytest -q` — **40 passed, 1 skipped** (шаблон).
После переноса файлов Lab 1 и до изменения её настроек: **207 passed, 2 skipped**.
Skipped — опциональные Docker/PostgreSQL-проверки. Python: 3.12.10, окружение
проекта `.venv`, определено через PyCharm. Реальные ключи и рабочий `.env` не читались.

Перенос выполнен только в рабочее дерево: `git restore --source=lab-1 --worktree`
для списка отличающихся файлов. Ветка, HEAD и индекс не переключались, коммиты и
теги не создавались. Рабочая БД не мигрировалась; проверялись отдельные тестовые БД.

## Архитектурные решения

Сохраняется один aiogram-бот, один конфигурационный модуль и существующая PostgreSQL.
Обработчики и LLM-клиент перенесены из Lab 1 без изменения поведения. Новый пакет
`app/agent` содержит модели, часы, порты внешних систем, счётчик вызовов, источник
расписания и репозиторий. Подключение к handlers и runtime оставлено следующему этапу.

Использован [официальный Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
`mcp==2.3.0` без CLI extras. Совместимость подтверждена установкой в существующее
окружение и импортом SDK в тесте точки входа. `python -m app.mcp_server` запускает
stdio-каркас через `MCPServer.run(transport="stdio")`; tools и resources пока не
зарегистрированы. Модуль не загружает токены/рабочую конфигурацию и не открывает БД.
Команда запуска по умолчанию использует тот же `sys.executable`; аргументы заданы
JSON-массивом без shell. `stdio_launch` готовит явный allowlist окружения.

Добавлены `tzdata==2026.5` для Windows и воспроизводимых IANA-зон и прямая декларация
уже используемого `pydantic==2.13.5`. Транзитивные зависимости SDK зафиксированы в
constraints. В стандартных тестах нет настоящего MCP-сеанса или внешнего HTTP.

`Lab2Settings` встроен в `Settings`: источник и зона расписания, команда/args/cwd MCP,
тайм-ауты, лимит 1–2 вызова, TTL 1–300 секунд, максимум один повтор погодного чтения.
`LAB2_ENABLED=false`; флаг зарезервирован для следующего этапа, сейчас его включение
не подключает агента. Пути относительны корню проекта. Пользовательская timezone не
выводится из зоны расписания. Настройки облачного запуска Lab 2 пока не подключены.

Расписание `data/schedule.json` синтетическое: 12–25 октября 2026, 14 дней с пустыми
выходными и занятиями `{start,end,title,location}`. Нет преподавателей, студентов,
личных адресов и групп. Неделя считается по внедрённым часам в заданной зоне;
файл вне этих дат даёт пустой список. Docker/архив включают только этот JSON.

Погода будет использовать [Open-Meteo](https://open-meteo.com/en/docs): для учебного
некоммерческого доступа API-ключ не требуется; одноимённые города требуют выбора.
Реальный адаптер и анализ его
ответов не реализованы, экспериментальных данных о погоде пока нет.

## Постоянные сущности и защита от дублей

Миграция `002_agent_foundation.sql` добавляет timezone к `user_settings`. Существующие
telegram_users, mode, temperature и conversation_messages не пересоздаются.

| Сущность | Назначение и ограничения |
|---|---|
| processed_updates | update_id PK, owner FK, processing/done/failed; подготовка и done атомарны |
| prepared_actions | неизменяемые нормализованные данные, state, сроки; UNIQUE(owner, source_update_id) |
| reminders | UNIQUE(action_id), UNIQUE(owner, idempotency_key), составной FK на действие того же owner |
| agent_audit | разрешённые коды действия/статуса/причины/ошибки, даты, время, длина текста; без произвольного текста |

Ключ идемпотентности равен UUID подготовленного действия и создаётся приложением.
Подготовка блокирует строку update; конкурирующий повтор возвращает существующее
действие. Повтор с изменённым payload отклоняется. Confirm/execute блокируют действие
через SELECT FOR UPDATE. INSERT reminder и переход executed выполняются атомарно;
повтор возвращает прежний ID с created=false, в том числе после открытия нового пула.

Триггеры БД запрещают изменение payload, повторное открытие terminal-состояния,
вставку напоминания без confirmed, подмену владельца/данных/ключа. Составные внешние
ключи защищают связи владельцев. Методы требуют owner, запросы всегда его фильтруют.
Это защита приложения и целостности связей; PostgreSQL RLS и отдельные роли здесь не
добавлены. Прямой доверенный администратор БД не входит в модель угроз.

Срок проверяется при confirm и execute; `get_action` показывает вычисленное expired,
а переход сохраняет его. Отменённое действие не исполняется. Новые текст/время
требуют нового update/action. Пока общий middleware всех update и callback не
подключён: реализована транзакционная дедупликация подготовки напоминания.

Авторизация владельца через реальный MCP пока не реализована: изменяющий инструмент
не опубликован. В следующем этапе нужен защищённый контекст/capability, который
независимый клиент не может сформировать из аргументов модели.

Репозиторий Lab 2 не читает историю диалогов и не логирует payload. Текст напоминания
возвращается только для preview владельца, скрыт из repr и не попадает в аудит.
Метаданные напоминания и последнего аудита не раскрывают внутренние owner/request ID.
Полная история Lab 1 остаётся в существующем репозитории для контекста LLM, как требует
сохранение поведения первой лабораторной.

## Проверки и результаты

Промежуточные проверки:

- 26 доменных тестов: даты/время, timezone, DST, строгие аргументы, лимит, расписание.
- 53 теста моделей, конфигурации и интерфейса репозитория — успешно.
- 4 новых PostgreSQL-теста — успешно: обновление 001→002, откат ошибочной миграции,
  конкуренция, изоляция, отмена/TTL, аудит и ограничения на прямой SQL.

Docker изначально был выключен; запущен локальный Desktop. Использован уже имеющийся
образ postgres:16-bookworm. Фикстура теперь задаёт `--pull=never`: тест не скачивает
образы. Каждый тест создаёт и удаляет только свой контейнер/volume, использует
loopback PostgreSQL, случайные реквизиты и синтетические ID. Telegram, LLM и погодных
сетевых запросов нет. Unit-тесты мигратора используют подменённое соединение.

Итоговые проверки (23.10.2026, Python 3.12.10):

| Команда | Фактический результат |
|---|---|
| `python -m pytest -q tests/test_agent_domain.py tests/test_agent_config.py tests/test_agent_repository.py tests/test_migrations.py tests/test_scripts.py` | **75 passed** |
| `RUN_INTEGRATION=1 python -m pytest -q tests/test_integration.py` | **6 passed**, включая 2 существующих теста Lab 1 |
| `python -m pytest -q` | **262 passed, 6 skipped**, 73.69 с |
| `python -m ruff check .` | **All checks passed!** |
| `python -m ruff format --check .` | **60 files already formatted** |
| `python -m pip check` | **No broken requirements found.** |
| `git diff --check` | Нет ошибок whitespace |

Шесть skipped в обычном прогоне — те же опциональные PostgreSQL-тесты, которые
отдельно прошли с RUN_INTEGRATION=1. Образ не скачивался; временных контейнеров и
volume с префиксом itmo-test- после прогона не осталось. Docker Desktop оставлен
работающим. В тестах ошибки Ruff (формат строк и resolve внутри async-теста)
исправлены до итогового прогона; падений функциональных проверок не было.

Дополнительно сверены Git-хеши с lab-1: entrypoint, handlers/study, history,
user_settings, llm/client, middleware, prompts и миграция 001 совпадают побайтово
после нормализации Git. HEAD остаётся df07a8d; в индексе по-прежнему только два
исходных файла пользователя. Большинство новых файлов намеренно untracked —
команда git add, коммит и тег не выполнялись.

## Соответствие обязательным сценариям методички

| № | Состояние на первом этапе |
|---|---|
| 1–3: погода/ошибки | Контракт и порт; адаптер и тесты следующего этапа |
| 4–6: расписание/пустой день/дата | Проверено на уровне сервиса и моделей |
| 7–11: подтверждение, валидация, дубли, изоляция | Проверено на уровне репозитория и настоящей PostgreSQL; Telegram/MCP ещё не подключены |
| 12: границы недели | Проверены граница года, месяца и смена дня в локальной зоне |
| 13: IANA и локальное время | Проверены неизвестная/отсутствующая зона, DST gap/fold |
| 14: неизвестный tool/лишние аргументы | Проверено на уровне строгого валидатора; discovery allowlist будет добавлен |

## Негативные сценарии полной Lab 2

| Сценарий | Ожидаемое поведение | Фактический статус |
|---|---|---|
| Неизвестный инструмент / раскрытие промпта | Отказ, никаких вызовов/раскрытия | Валидатор unknown_tool проверен; диалоговый сценарий не запускался |
| Инструкция внутри результата инструмента | Данные не изменяют правила агента | Проверено агентным циклом: результат передаётся как tool data |
| Неоднозначная дата / прошлое напоминание | Уточнение/отказ, ноль записей | Проверено сервисом и Telegram-границей с фиксированными часами |
| Недоступность MCP / погодного API | Понятная ошибка без ложного успеха | Проверено fault-тестами клиента, агента и handlers |

## Эксперимент маршрутизации (заполнить на этапе 4)

Исходный набор `docs/evaluation/lab2-routing.json` сохранён без изменений: 12 запросов,
фиксированные время/зона. Эксперимент с реальной моделью **не проводился**.
Неизвестны фактические действия/аргументы; routing_accuracy и argument_accuracy
не вычислялись. Требуется таблица всех 12 случаев (ожидание, факт, аргументы,
результат), версия модели и настройки; ≥10/12 и ≥8/9; анализ ошибок, одно изменение
и повтор ошибочных запросов с сохранением обеих версий. Никакие fake-ответы не
выдаются за экспериментальные результаты.

## Что оставалось после этапа 1

Три действующих MCP-инструмента, weekly resource через MCP, discovery, доверенный
контекст владельца, реальный weather adapter, агентный цикл и команды
`/tools`/`/timezone`/`/why`, inline-кнопки, общий учёт update/callback, обработка
отказов на границах, независимый клиент и эксперимент. Рассылка напоминаний — Lab 3.
Тег lab2 из критериев сдачи сейчас намеренно не создаётся по инструкции пользователя.

## Изменённые файлы

Собственно фундамент Lab 2:

- `AGENTS.md`, `spec.md`, `plan.md`, `docs/lab2-report.md`, дополнение `README.md`.
- `app/config.py`, `.env.example`, `requirements.txt`, `constraints.txt`.
- `app/agent/__init__.py`, `errors.py`, `clock.py`, `models.py`, `ports.py`,
  `budget.py`, `schedule.py`, `repository.py`, `mcp_launch.py`.
- `app/mcp_server/__init__.py`, `app/mcp_server/__main__.py`.
- `migrations/002_agent_foundation.sql`, `data/schedule.json`.
- `Dockerfile`, `.dockerignore`, `scripts/cloud.py`: упаковка синтетического JSON.
- `tests/test_agent_domain.py`, `test_agent_config.py`, `test_agent_repository.py`,
  `test_migrations.py`, дополнения `test_integration.py`, `test_scripts.py`.

Также перенесены файлы готовой Lab 1, которых не было в HEAD lab-2:
изменения app entrypoint/config/db/health/logging/telegram, handlers/study,
history, llm, middleware, prompts, user_settings; миграция 001 и scripts/migrate;
README, конфигурация сборки/зависимостей/Compose, local/cloud scripts;
существующие тесты app/compose/scripts/integration и context_config/experiment/
history/llm/modes/telegram/user_settings; docs/demonstration, lab1-report,
experiment-results и scripts/experiment. Это исходная работа, а не новая реализация
Lab 2. Уже staged-файлы методички и routing-набора принадлежат пользователю.

## Этапы 2–3: MCP и интеграция чтения, 24.10.2026

Предыдущая работа остановилась после записи MCP-модулей, до их проверки. Исходный
прогон этого продолжения: **260 passed, 6 skipped, 2 failed**. Обе ошибки были в
устаревших тестах каркаса: перечень env-полей и вызов entrypoint SDK. Они обновлены
под настоящий сервер; проверки функциональности Lab 1 не удалены.

### Реализованный путь запроса

1. `app.__main__.run` открывает `agent_runtime` в той же задаче, в которой закрывает
   MCP-сессию. Стандартный stdio, официальный SDK, discovery tools/resources.
   При отказе старта сохраняется безопасный код в техническом аудите, polling
   всё равно начинается. Бот запускает ребёнка без Telegram/LLM/DB-секретов.
2. Существующий handler выбирает режим/temperature, сохраняет сообщение и берёт
   ограниченную историю Lab 1. Доверенный owner — `message.from_user.id`.
   Handler не содержит MCP JSON-схем, SQL Lab 2 или погодного HTTP-клиента.
3. Внедряемый `AgentLLM` сначала возвращает строгий Intent по истории и текущему
   обнаруженному каталогу. При потребности в актуальных данных native вызов
   использует `tool_choice=required`. Произвольный текст не разбирается как команда.
4. Native Responses function calling получает схемы из MCP-каталога. В auto-режиме
   только явная ошибка неподдерживаемых tool-параметров включает строгий Pydantic
   structured fallback. Для провайдеров без native tools есть конфигурация
   `LAB2_LLM_TOOL_MODE=structured`; поддержка strict JSON остаётся обязательной.
5. `ReadAgent` проверяет имя, исходное намерение, доменную модель и обнаруженную
   inputSchema; расходует CallBudget до вызова. Валидирует outputSchema, доменную
   модель результата и соответствие даты расписания запросу.
6. Native tool-result передаётся как `function_call_output` с исходным call_id.
   Fallback использует JSON-конверт `tool_result` в пользовательской роли.
   Ни один результат не дописывается в системные инструкции. Исходный набор
   разрешённых действий не расширяется из результатов инструментов.
7. После максимум двух MCP-вызовов следующая попытка останавливается с сообщением
   и `limit_exceeded` в аудите. Итог проходит прежнюю отправку длинных ответов и
   сохранение assistant-истории. На каждое сообщение отдельный бюджет и request ID.

Поток native tool calling сверялся с
[официальной документацией Responses](https://developers.openai.com/api/docs/guides/function-calling).
Обычный диалог теперь использует два LLM-запроса (intent и ответ), чтение — до
четырёх, без учёта попытки native перед явным fallback. Это цена отдельной проверки
потребности в актуальных данных, в том числе при недоступном MCP. При отключённом
`LAB2_ENABLED` используется исходный Lab 1-поток с одним generate.

### MCP, изменения и безопасность

Обнаруживаются get_weather/get_schedule/add_reminder и schedule://current-week.
Все инструменты имеют реальные inputSchema/outputSchema и structuredContent.
Weather использует фиксированные Open-Meteo endpoints, timeout, максимум один
повтор GET и ограничение тела ответа; неоднозначные города возвращаются как
различимые варианты, первый автоматически не выбирается. Расписание читается
только из настроенного JSON; неделя строится через Clock.

Для изменения сервер проверяет HMAC capability из служебных метаданных. Подписаны
owner, action ID, хеш аргументов, audience и срок не более 300 с. Репозиторий
повторно сверяет аргументы под блокировкой строки. Транзакция выполняет действие
один раз, повтор возвращает тот же ID с `created_new=false`.
На этапе чтения Telegram ещё не выпускал capability. Текущий этап заменил временный
`DeferredReminderPreparation` на `ReminderService`: capability выпускается только после
проверки владельца и явного inline-подтверждения.

`/tools` показывает каталог подключённого клиента, описание и read/write-аннотации;
список не зашит в handler. `/start` дополнен этой командой, `/timezone` и `/why`.
Конфигурация Lab 2 передаётся
также через Compose; образ уже включает синтетический data/schedule.json.

Аудит использует существующую agent_audit: время, случайный request ID, закрытые
action/tool/status/reason/error, duration_ms и типизированное резюме. Для города
сохраняется только длина в text_length, для расписания — дата, для предложения
напоминания — длина и время. Полных сообщений, городов, токенов, системных инструкций,
скрытых рассуждений и сырого текста исключений в аудите нет. Owner — отдельный FK,
не текстовый лог. SDK-логи подавлены фильтром, в том числе при DEBUG, чтобы SDK
не печатал тела запросов/ответов и служебные метаданные. Диагностика границ —
контролируемые app.* события. Для подтверждений добавлена новая миграция 003;
применённые 001/002 не изменены.

### Фактические проверки

Все команды запускались из корня с Python 3.12.10, выбранным в PyCharm:
`.\.venv\Scripts\python.exe`. Рабочие .env и реальные ключи не читались;
Telegram, LLM и погода в тестах подменены. SDK-протокол проверяется в памяти,
настоящий stdio-процесс включается отдельным RUN_MCP.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_read_agent.py tests/test_agent_llm.py tests/test_mcp_protocol.py tests/test_capability.py tests/test_weather.py tests/test_agent_config.py tests/test_compose.py
# 102 passed in 5.02s

.\.venv\Scripts\python.exe -m pytest -q
# 344 passed, 8 skipped in 19.11s

.\.venv\Scripts\python.exe -m ruff check .
# All checks passed!
.\.venv\Scripts\python.exe -m ruff format --check .
# 75 files already formatted
.\.venv\Scripts\python.exe -m pip check
# No broken requirements found.

$env:RUN_INTEGRATION='1'
try { & .\.venv\Scripts\python.exe -m pytest -q -m integration } finally { Remove-Item Env:RUN_INTEGRATION }
# 7 passed in 37.58s

$env:RUN_MCP='1'
try { & .\.venv\Scripts\python.exe -m pytest -q -m mcp } finally { Remove-Item Env:RUN_MCP }
# 1 passed in 7.91s
```

Восемь штатных skip — семь PostgreSQL и один stdio-тест без opt-in флагов.
PostgreSQL-тест дополнительно проверил четыре конкурентных MCP-запроса на одно
подтверждённое действие: ровно одна запись и один created_new=true, одинаковый ID
для всех. Прямой вызов без capability и изменение payload отклонены. Агент чтения
через настоящий протокол сохранил безопасный аудит в тестовую PostgreSQL.
После тестов контейнеров и volume с префиксом itmo-test- не осталось.
Stdio smoke обнаружил три инструмента/ресурс, прочитал занятия и неделю; погодный
HTTP не вызывался. `git diff --check` прошёл. Коммиты, теги и git add не выполнялись.

| Проверяемое требование | Доказательство |
|---|---|
| Обычный ответ, история, режимы, температура | test_read_agent + исходные test_modes/test_llm/test_history |
| Погода, расписание, пустой день, строгие аргументы | test_read_agent/test_mcp_protocol/test_weather |
| Native schemas, output linkage, strict fallback | test_agent_llm |
| Попытка инструкции в tool-result | Данные остаются в exchange, отсутствуют в instructions; новый add_reminder не разрешается |
| Третий MCP-вызов | Отказ до вызова, ровно два обращения, audit limit_exceeded |
| MCP отсутствует | Polling запускается; обычный ответ работает; запрос данных получает отказ |
| /tools из discovery | Тестовый нестандартный каталог отображается без статического списка |
| Нет утечек в аудите | Проверки sentinel-строк, owner в текстовых логах и подавления SDK payload |
| Capability / повтор / границы недели | test_capability, MCP+PostgreSQL, даты через месяц/год |

### Файлы этапов 2–3

- MCP: `app/mcp_server/server.py`, `app/mcp_server/__init__.py`,
  `app/mcp_server/__main__.py`; `app/agent/mcp_client.py`, `mcp_contracts.py`,
  `capability.py`, `weather.py`, `mcp_launch.py`.
- Агент и существующий бот: `app/agent/decisions.py`, `service.py`, `runtime.py`,
  `models.py`, `repository.py`; `app/llm/agent.py`, `app/llm/client.py`,
  `app/handlers/study.py`, `app/__main__.py`, `app/logging_setup.py`.
- Конфигурация/зависимости: `app/config.py`, `.env.example`, `compose.yaml`,
  `requirements.txt`, `pyproject.toml`. aiohttp/jsonschema объявлены прямыми
  зависимостями, поскольку теперь импортируются приложением; уже были в окружении.
- Тесты: новые `test_read_agent.py`, `test_agent_llm.py`, `test_mcp_protocol.py`,
  `test_mcp_stdio.py`, `test_capability.py`, `test_weather.py`; обновлены
  `test_agent_config.py`, `test_integration.py`, `test_compose.py`.
- Документация: `AGENTS.md`, `spec.md`, `plan.md`, `README.md`, этот отчёт.

## Этап 4: безопасное подтверждение напоминаний, 24.10.2026

Миграция `003_reminder_confirmation.sql` добавляет вычисляемый fingerprint действия и
привязку callback-update к владельцу, действию и операции. Подготовка обычного сообщения
и callback сериализуются advisory lock в отдельном coordination pool; основная DB-сессия
остаётся доступна MCP-процессу. Ограничения 002/003 и транзакция `execute_confirmed`
гарантируют одну запись при повторной доставке и параллельном подтверждении.

`/timezone` принимает только IANA-зону через `ZoneInfo`, сохраняет её и показывает
фиксируемое через Clock локальное время. Предложение модели имеет строгую схему только
из `text` и локального `YYYY-MM-DDTHH:MM[:SS]`. Приложение само применяет сохранённую
зону, отклоняет прошлое время, DST gap/fold, лишние поля и отсутствие зоны. Действие
живёт не более пяти минут и показывает inline-кнопки подтверждения и отмены.

При callback проверяются Telegram owner, приватный чат, UUID и fingerprint. Capability
с owner/action/idempotency передаётся MCP вне аргументов модели. Успешный ответ сверяется
с записью PostgreSQL до показа ID, точного времени и зоны. Отмена и повтор исполнения
идемпотентны. `/why` читает последнюю запись только текущего владельца и отображает
типизированные безопасные поля без исходного сообщения, owner ID, промптов и секретов.

Unit-тесты используют fake LLM/MCP/репозитории и фиксированные часы. Отдельный тест с
настоящим временным PostgreSQL проверяет четыре конкурентные подготовки одного update,
двойное подтверждение и повтор callback: получается одно действие и одно напоминание.

Финальные проверки этапа:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# 368 passed, 9 skipped in 21.08s

$env:RUN_INTEGRATION='1'
.\.venv\Scripts\python.exe -m pytest -m integration -q
# 8 passed, 369 deselected in 35.56s
Remove-Item Env:RUN_INTEGRATION

$env:RUN_MCP='1'
.\.venv\Scripts\python.exe -m pytest -m mcp -q
# 1 passed, 376 deselected in 8.09s
Remove-Item Env:RUN_MCP

.\.venv\Scripts\python.exe -m ruff check .
# All checks passed!
.\.venv\Scripts\python.exe -m ruff format --check .
# 78 files already formatted
.\.venv\Scripts\python.exe -m pip check
# No broken requirements found.
```

### Оставшиеся этапы

Нет автоматического переподключения после отказа MCP: для восстановления соединения
нужен перезапуск бота. Отдельный эксперимент маршрутизации 12 запросов с настоящей
фиксированной моделью и измерением точности не проводился; fake-тесты его не заменяют.
Inspector вручную не запускался. Рассылка наступивших напоминаний относится к Lab 3.
