from mlpynem.simulation import SpectrumSimulation
import numpy as np
from tqdm import tqdm
import itertools

class Dataset():
    """
    Dataset class for the PINEM model. It used to generate a set of PINEM spectra for the training of a neural network.
    The dataset can be saved and loaded from a file.

    Instanciation Args:
    - x (np.ndarray): The energy axis of the spectra
    - amplitude (float): The amplitude of the spectra
    - kernel: Shape of the Zero Loss Peak (ZLP) (np.ndarray or str). The possible strings are 'Gaussian' or 'Voigt'
    - Parameters for the PINEM model
        - lower_bound (np.ndarray): The lower bound of the parameters
        - upper_bound (np.ndarray): The upper bound of the parameters
    - n (int): The number of spectra to generate
    - load (str): The path to the file to load the dataset from. When this argument is used, the other arguments are ignored.
    - seed (int): The seed for the random number generator
    - n_cutoff (int): The cutoff for the sidebands of the PINEM model
    - background (float): The background level in percents of the amplitude
    """
    def __init__(
        self,
        simulation : SpectrumSimulation = None,
        parameters : dict = None,
        amplitude: float = 1.0,
        n: int = 1024 * 64,
        load: str = None,
        seed: int = 42,
        background: np.array = 0.0
    ):
        if load is None : 
            self.n = n
            self.parameters = parameters
            self.seed = seed
            self.amplitude = amplitude
            self.simulation = simulation
            self.background = background
            # self.gen_data()            
        else : 
            self.load_model(load)

    # def expand_bounds(self) : 
    #     """
    #     Expand the bounds of the parameters to the number of spectra to generate
    #     """

    #     lb = np.expand_dims(self.lower_bound, axis=1)
    #     ub = np.expand_dims(self.upper_bound, axis=1)
    #     nlb = lb@(np.ones((1,self.n)))
    #     nub = ub@(np.ones((1,self.n)))
    #     return nlb, nub
            
    def generate_truth(self, method : str = 'random') :
        """
        Generate the true values of the parameters for the dataset. The method is 'random' for now.
        """        
        if method == 'random' :
            self.truth = self.init_truth(self.n)
            for key in self.parameters.keys() :
                self.truth[key] = np.random.uniform(self.parameters[key][0], self.parameters[key][1], size=(self.n,))
        elif method == 'grid' :
            new_params = {}
            actual_n = 1
            for key in self.parameters.keys() :
                if self.parameters[key][1] - self.parameters[key][0] == 0 :
                    new_params[key] = np.array([self.parameters[key][0]])
                else :
                    new_params[key] = np.linspace(self.parameters[key][0], self.parameters[key][1], self.n)
                    actual_n *= self.n
            self.truth = self.init_truth(actual_n)
            for j, params in enumerate(iterate_combinations(new_params)):
                for key in params.keys() :
                    self.truth[key][j] = params[key]
            self.n = actual_n

        else :
            raise ValueError("The method must be 'random' or 'grid'")
        
    def init_truth(self, length : int = None) :
        list_dt = [(key, np.float64, length) for key in self.parameters.keys()]
        # return np.array([np.zeros()]*len(list_dt), dtype = np.dtype(list_dt))  
        return np.zeros((), dtype = np.dtype(list_dt))      

    def gen_data(self) :
        """
        Generate random values for the parameters and generate the spectra. Stores both the noisy and noiseless spectra in the object.
        """
        np.random.seed(self.seed)
        # One set of random parameters for each spectrum
        len_energy = self.simulation.energy_axis.shape[0]
        spectres = np.zeros((self.n, len_energy))
        for i in tqdm(range(self.n)) :
            current_dict = {key : self.truth[key][i] for key in self.parameters.keys()}
            spectres[i,:] = self.simulation.output_spectrum(**current_dict)[:,np.newaxis].T 
        # The spectres are normalized to the range [0, self.amplitude] then we add the background 
        self.noiseless_spectres = (spectres + self.background)*self.amplitude
        self.noisy_spectres = np.random.poisson((spectres+self.background)*self.amplitude)
            

    def __len__(self):
        return self.n
    
    # TODO : Add a __getitem__ method to get a specific spectrum

    # TODO : Add a gen_data_vec, that generates the data in a vectorized way

    def get_truth(self, names) :
        """
        Get the true values of some of the parameters for the dataset.

        Args:
        - names (list[str]): The names of the parameters to get
        """
        l = [self.truth[name] for name in names]
        ar = np.array(l).T
        return ar
    
    def save_model(self, path) :
        """
        Save the dataset to a file.

        Args:
        - path (str): The path to the file to save the dataset to
        """
        d = {}
        d['amplitude'] = self.amplitude
        d['seed'] = self.seed
        d['n'] = self.n
        d['background'] = self.background
        d['parameters'] = {}
        d['simulation'] = {}
        for key in self.parameters.keys() :
            d['parameters'][key] = self.parameters[key]
        for key in self.simulation.as_dict().keys() :
            d['simulation'][key] = self.simulation.as_dict()[key]
        d['truth'] = self.truth
        d['noiseless_spectres'] = self.noiseless_spectres
        d['noisy_spectres'] = self.noisy_spectres
        np.savez(path, **d)

    def load_model(self, path) :   
        """
        Load the dataset from a file. The file must have been saved with the save_model method.When called this method bypasses the generation of the dataset.

        Args:
        - path (str): The path to the file to load the dataset from.
        """
        d = np.load(path, allow_pickle=True)
        self.truth = d['truth']
        self.noisy_spectres = d['noisy_spectres']
        self.noiseless_spectres = d['noiseless_spectres']
        self.background = d['background']
        self.seed = d['seed']
        self.n = d['n']
        self.amplitude = d['amplitude']
        
    
def normalize_spectra(spectra: np.ndarray) -> np.ndarray:
    """
    Normalize a set of spectra to the range [0, 1]

    Args:
    - spectra (np.ndarray): The set of spectra with dimension (number of spectra, spectra length) to normalize
    """
    m = np.min(spectra, axis=1)
    M = np.max(spectra, axis=1)
    int_sp = (spectra - m[:, np.newaxis]) / (M - m)[:, np.newaxis]
    if np.any(np.isnan(int_sp)) : 
        print('nan')
        print(np.argwhere(np.isnan(int_sp)))
    return (spectra - m[:, np.newaxis]) / (M - m)[:, np.newaxis]

def iterate_combinations(params_dict):
    """
    Helper function to iterate through all the combinations of the parameters in params_dict.
    
    Args:
    - params_dict (dict): A dictionary of lists with the parameters to vary. Lists of floats or integers are expected.
    """
    keys = params_dict.keys()
    values = params_dict.values()
    combinations = list(itertools.product(*values))
    for combination in combinations:
        yield dict(zip(keys, combination))

