class ForceCalculator:
    """Базовый класс для калькуляторов сил."""
    def calculate(self):
        raise NotImplementedError("Метод calculate должен быть реализован в подклассе.")

    def calculate_forces_and_convection(self, positions, radii):
        raise NotImplementedError("Метод calculate_forces_and_convection должен быть реализован в подклассе.")
