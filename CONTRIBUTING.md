# Вклад в проект

Спасибо за интерес к **particle-simulator**. Прежде чем открыть issue или
pull request, ознакомьтесь с короткими правилами ниже.

## Сообщение об ошибке

Открывайте [issue](https://github.com/ddkx21/particle-simulator/issues)
с минимальным воспроизведением:

- версия Python (требуется 3.12),
- версии `numpy`, `scipy`, `taichi` (`pip freeze | grep -E "numpy|scipy|taichi"`),
- ОС и архитектура,
- команда запуска и сообщение об ошибке (полный traceback),
- по возможности — короткий скрипт (≤ 30 строк), воспроизводящий проблему.

## Pull Request

1. Форкните репозиторий, создайте ветку от `master`:
   ```bash
   git checkout -b feature/my-change
   ```
2. Установите окружение разработки:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Перед коммитом прогоните проверки:
   ```bash
   ruff check dem/ pbm/ tests/
   black --check dem/ pbm/ tests/
   pytest tests/ --cov=dem --cov=pbm
   ```
4. Сообщения коммитов — в стиле
   [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`,
   `fix:`, `refactor:`, `docs:`, `test:`, `chore:`).
5. Откройте PR в `master`. CI запустит lint + tests + build на Python 3.12.

## Стиль кода

- Форматтер: `black` (line-length = 100).
- Линтер: `ruff` (правила в `pyproject.toml`).
- Типы: `mypy` (warn-only в CI; постепенно ужесточаем).
- Тесты: `pytest`, цель — покрытие 80 %+ для новых модулей.

## Контакты

По вопросам, не подходящим для публичного issue:

- Email: <rustam.dadakhodjaev@gmail.com>
- Учебная почта (СПбГУ): <st094266@student.spbu.ru>
- Telegram: [@ddkx21](https://t.me/ddkx21)

— **Dadakhodjaev Rustam Bakhtiyorovich**
