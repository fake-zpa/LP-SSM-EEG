"""Training callbacks (placeholder — logic integrated into Trainer)."""


class BaseCallback:
    def on_epoch_end(self, epoch: int, metrics: dict):
        pass

    def on_train_end(self, summary: dict):
        pass
