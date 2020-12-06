from simulators import DataSimulator
from concurrent.futures import ThreadPoolExecutor

class Validator:
    def __init__(
        self,
        trained_model,
        specifications: dict,
        simulator: DataSimulator,
    ):
        for name in simulator.interactions.keys():
            assert hasattr(trained_model, name)
            assert callable(trained_model.name)
        self.simulator = simulator
        self.specifications = specifications
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
                size = size_sims + [size_sims[-1] * (n_sims - len(size_sims))]
        else:
            size = [size_sims] * n_sims
        if multithreaded:
            with ThreadPoolExecutor(max_workers=None) as executor:
                result = list(executor.map(self, size))
        else:
            result = [self(size[i]) for i in range(n_sims)]
        return result

    def _validate(self, size):
        pass

    def _validate_metrics(self):
        pass

    def _validate_derivatives(self):
        pass