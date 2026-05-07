class ParticleGenerator:

    # Генерирует частицы и возващает объект ParticleState
    def generate(self, count):
        """
        Генерация частиц.

        :param count: Количество частиц.
        :return: Объект ParticleState с генерированными частицами.
        """
        raise NotImplementedError("Метод generate должен быть реализован в подклассе.")
    
    def save(self, filename):
        raise NotImplementedError("Метод save должен быть реализован в подклассе.")
    
    def load(self, filename):
        raise NotImplementedError("Метод load должен быть реализован в подклассе.")
    
    def plot(self):
        raise NotImplementedError("Метод plot должен быть реализован в подклассе.")