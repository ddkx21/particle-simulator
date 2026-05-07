class Solver:
    """Базовый класс для решателей задачи."""

    def __init__(self, particle_generator, force_calculator, solution, post_processor):
        self.particle_generator = particle_generator
        self.force_calculator = force_calculator
        self.solution = solution
        self.post_processor = post_processor

    def solve(self, dt, total_steps):
        raise NotImplementedError("Метод solve должен быть реализован в подклассе.")
