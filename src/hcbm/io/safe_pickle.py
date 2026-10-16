"""Restricted unpickling: only numpy array reconstruction is allowed."""
import pickle


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "numpy":
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_numpy_pickle(path, encoding="latin1"):
    with open(path, "rb") as f:
        return _NumpyOnlyUnpickler(f, encoding=encoding).load()
