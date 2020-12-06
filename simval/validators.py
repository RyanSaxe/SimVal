from simulators import DataSimulator

class Validator:
    def __init__(
        self,
        trained_model,
        simulator: DataSimulator,
    ):
        for name,interaction in simulator.interactions.items():
            assert hasattr(trained_model, name)
            assert callable(trained_model.name)
        self.interactions = simulator.interactions
        self.model = trained_model

    def __call__(self, *args):
        # concept taken from: https://github.com/mdbloice/Augmentor/
        #   writing multiprocessing to pass through the callable
        #   part of the class makes it possible for the procedure 
        #   to be *pickled* and therefore suitable to multithreading
        #   . . . do not call this directly
        return self._validate(*args)

    def validate(self, n_sims, size_sims, multithreaded=True):
        if isinstance(size_sims, list):
            if n_sims > len(size_sims):
                size_sims = size_sims + [size_sims[-1] * (n_sims - len(size_sims))]
        else:
            size_sims = [size_sims] * n_sims
        if multithreaded:
            with ThreadPoolExecutor(max_workers=None) as executor:
                result = list(executor.map(self, [size] * n_sims))
        else:
            result = [self(size) for i in range(n_sims)]

    def _validate(self, size):
        pass