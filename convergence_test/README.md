# Convergence Test: Pseudo-Periodicity

Тест сходимости псевдо-периодического метода (MIC + COMSOL correction)
к истинно-периодическому решению (суперячейка K³ копий).

## Запуск

```bash
python -m convergence_test.convergence_runner
```

## Результаты

Сохраняются в `convergence_test/results/`:
- `convergence_results.csv` — таблица ошибок
- `convergence_*.png` — графики сходимости
