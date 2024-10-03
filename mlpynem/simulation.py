from abc import ABC, abstractmethod
import numpy as np

class SpectrumSimulation(ABC):
    def __init__(self, energy_axis : np.ndarray) :
        self.energy_axis = energy_axis
        self.scale = self.energy_axis[1] - self.energy_axis[0]

    @abstractmethod
    def simulation(self, *args, **kwargs):
        pass

    def normalize(self, spectrum, amplitude):
        m = np.min(spectrum)
        M = np.max(spectrum)
        return amplitude*(spectrum - m)/(M - m)

    def output_spectrum(self, *args, amplitude : float = 1.0, **kwargs):
        spectrum = self.simulation(*args, **kwargs)
        return self.normalize(spectrum, amplitude)
    
    @abstractmethod
    def simulation_tips(self):
        pass
    
    def as_dict(self) : 
        attr_names = dir(self)
        attr_dict = {}
        for attr_name in attr_names:
            if not callable(getattr(self, attr_name)) and not attr_name.startswith("__") and not attr_name.startswith("_") :
                attr_dict[attr_name] = getattr(self, attr_name)
        return attr_dict
