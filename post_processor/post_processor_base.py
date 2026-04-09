class PostProcessor:
    """Базовый класс для постобработки и визуализации результатов."""

    def __init__(self, solution):
        self.solution = solution

    def plot(self):
        raise NotImplementedError("Метод plot должен быть реализован в подклассе.")

    def live_plot(self):
        raise NotImplementedError("Метод live_plot должен быть реализован в подклассе.")

    def initialize_live_plot(self):
        raise NotImplementedError("Метод initialize_live_plot должен быть реализован в подклассе.")

    def finalize_live_plot(self):
        raise NotImplementedError("Метод finalize_live_plot должен быть реализован в подклассе.")

    def on_close_live_plot(self):
        raise NotImplementedError("Метод on_close_live_plot должен быть реализован в подклассе.")
