# stages.py
from abc import ABC, abstractmethod


class AnalysisStage(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the stage."""
        pass


class DataLoader(ABC):
    @abstractmethod
    def load_data(self):
        """Return (power_grid, currents, frequencies, settings) for a given readout type."""
        pass


class Fitter(ABC):
    @abstractmethod
    def fit(self, data):
        """Perform fitting on the provided data and return a result object or dictionary."""
        pass


class Plotter(ABC):
    @abstractmethod
    def plot(self, data):
        """Generate plots from the provided data."""
        pass
