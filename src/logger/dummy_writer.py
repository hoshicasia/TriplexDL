class DummyWriter:
    def __init__(
        self, logger=None, project_config=None, run_name="experiment", **kwargs
    ):
        self.run_name = run_name
        self.step = 0
        self.mode = "train"
        print(f"Initialized logging for experiment: {run_name}")

    def set_step(self, step, mode="train"):
        self.step = step
        self.mode = mode

    def add_scalar(self, scalar_name, scalar):
        pass

    def add_scalars(self, scalars):
        pass
