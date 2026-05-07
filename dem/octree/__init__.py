"""
Плоское октодерево (Flat Octree) для расчёта сил методом Барнса-Хатта.

Модуль предоставляет:
- OctreeNode: структура узла с bounding box, R³-агрегатами, force_sum
- FlatOctree: плоское дерево без рекурсии, stackless walk, multi-particle leaves
- TreeDropletForceCalculator: интеграция с force_calculator интерфейсом
- TreeStatistics / compute_tree_stats / print_tree_stats: расширенная диагностика дерева
- visualize_tree: интерактивная 3D-визуализация (PyVista, необязательная зависимость)

Параметры алгоритма:
- theta: параметр Барнса-Хатта (критерий аппроксимации "кластер-капля")
- mpl: max particles per leaf — контролирует глубину дерева
"""

from .flat_tree import FlatOctree
from .force_tree import TreeDropletForceCalculator
from .octree_node import OctreeNode
from .tree_stats import TreeStatistics, compute_tree_stats, print_tree_stats

try:
    from .tree_visualizer import visualize_tree
except ImportError:
    pass

__all__ = [
    "OctreeNode",
    "FlatOctree",
    "TreeDropletForceCalculator",
    "TreeStatistics",
    "compute_tree_stats",
    "print_tree_stats",
    "visualize_tree",
]
