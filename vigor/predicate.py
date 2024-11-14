class Predicate:
    def __init__(self, statistic: str, min: float, max: float) -> None:
        self.min = min
        self.max = max
        self.mu = (min + max) / 2
        self.a = 1 / ((max - min) / 2)
        self.statistic = statistic

    def relevance(self, value: float) -> float:
        """Returns a score based on the evaluation of the statistic."""
        if value < self.min or value > self.max:
            return 0
        return (value - self.min) / (self.max - self.min)