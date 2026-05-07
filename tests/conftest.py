"""Общие фикстуры для всего тестового набора DEM-симулятора.

- Гарантируем, что корень репозитория есть в sys.path (для `import pbm`,
  `import octree`, `import collision_detector` и т.д.).
- Однократная инициализация Taichi на CPU/f64 для всех тестов сессии.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def taichi_session():
    """Однократная инициализация Taichi на сессию.

    Все taichi-зависимые модули (octree, force_calculator, collision_detector,
    pbm.collision_frequency.tree_collision_freq) делят один runtime.
    """
    import taichi as ti
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    yield
    # Taichi не требует явного shutdown; следующий ti.init() пересоздаст runtime.
