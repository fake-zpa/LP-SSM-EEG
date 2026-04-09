"""Early stopping callback."""


class EarlyStopping:
    def __init__(self, patience: int = 15, mode: str = "max", min_delta: float = 0.001,
                 min_epochs: int = 0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best = None
        self.counter = 0
        self.should_stop = False
        self._epoch = 0

    def step(self, value: float) -> bool:
        self._epoch += 1
        if self._epoch <= self.min_epochs:
            if self.best is None or (
                (value > self.best) if self.mode == "max" else (value < self.best)
            ):
                self.best = value
            return False
        if self.best is None:
            self.best = value
            return False
        improved = (
            (value > self.best + self.min_delta) if self.mode == "max"
            else (value < self.best - self.min_delta)
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
