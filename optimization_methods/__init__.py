# optimization_methods/__init__.py
from .gradient_descent import GradientDescent
from .quadratic_simplex import QuadraticSimplex
from .genetic_algorithm import GeneticAlgorithm
from .particle_swarm import ParticleSwarmOptimization
from .bees_algorithm import BeesAlgorithm
from .immune_network import ImmuneNetworkOptimization
from .bacterial_foraging import BacterialForagingOptimization
from .hybrid_gapso import HybridGAPSO

optimization_methods = {
    "Градиентный спуск": GradientDescent,
    "Квадратичный симплекс": QuadraticSimplex,
    "Генетический алгоритм": GeneticAlgorithm,
    "Рой частиц": ParticleSwarmOptimization,
    "Алгоритм пчел": BeesAlgorithm,
    "Иммунная сеть": ImmuneNetworkOptimization,
    "Бактериальная оптимизация": BacterialForagingOptimization,
    "Гибридный GA-PSO": HybridGAPSO
}