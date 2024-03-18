from mlpynem.pinem import Pinem
import numpy as np
from tqdm import tqdm
import itertools

class DatasetPinem():
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
        x: np.ndarray,
        amplitude: float,
        kernel,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        n: int = 1024 * 64,
        load: str = None,
        seed: int = 42,
        n_cutoff: int = 50,
        background: float = 0.0
    ):
        if load is None : 
            self.n = n
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
            self.seed = seed
            self.amplitude = amplitude
            self.p = Pinem(x=x,amplitude=amplitude, kernel=kernel, n_cutoff=n_cutoff)
            self.background = background
            self.gen_data()            
        else : 
            self.load_model(load)


    def expand_bounds(self) : 
        """
        Expand the bounds of the parameters to the number of spectra to generate
        """
        lb = np.expand_dims(self.lower_bound, axis=1)
        ub = np.expand_dims(self.upper_bound, axis=1)
        nlb = lb@(np.ones((1,self.n)))
        nub = ub@(np.ones((1,self.n)))
        return nlb, nub

    def gen_data(self) :
        """
        Generate random values for the parameters and generate the spectra. Stores both the noisy and noiseless spectra in the object.
        """
        np.random.seed(self.seed)
        elb, eub = self.expand_bounds()
        # One set of random parameters for each spectrum
        self.xs = (np.random.uniform(elb, eub, size=(5,self.n))).T
        len_x = self.p.x.shape[0]
        self.spectres = np.zeros((self.n, len_x))
        for i, x in tqdm(enumerate(self.xs)) :
            self.spectres[i,:] = (self.p.calc_sq_modulus(omega=x[3], g =  x[0], offset=  x[4], fwhm = x[2], rt = x[1]))[:,np.newaxis].T 
        # The self.spectres are normalized to the range [0, self.amplitude] then we add the background  
        self.noiseless_spectres = self.spectres + self.amplitude*self.background
        self.noisy_spectres = np.random.poisson(self.spectres+self.background*self.amplitude)
            

    def __len__(self):
        return self.n
    
    # TODO : Add a __getitem__ method to get a specific spectrum
    
    def save_model(self, path) :
        """
        Save the dataset to a file.

        Args:
        - path (str): The path to the file to save the dataset to
        """
        d = {}
        float_dt = np.dtype([('amplitude', np.float64),
                             ('n_cutoff', np.int64),
                             ('n', np.int64),
                             ('background', np.float64),
                             ('seed', np.int64)])
        float_arr = np.array([(self.p.amplitude, self.p.n_cutoff, self.n, self.background, self.seed)], dtype=float_dt)
        bounds_dt = ['lower_bound', 'upper_bound']
        bounds_arr = np.rec.fromarrays([self.lower_bound, self.upper_bound], names=bounds_dt)
        d['x'] = self.p.x
        if type(self.p.kernel) == np.ndarray :
            d['kernel'] = self.p.kernel
        else : 
            str_arr = np.rec.fromarrays([self.p.kernel], names=['kernel'])
            d['kernel'] = str_arr
        d['floats'] = float_arr
        d['bounds'] = bounds_arr
        d['xs'] = self.xs
        d['spectres'] = self.spectres 
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
        self.xs = d['xs']
        self.spectres = d['spectres']
        self.noisy_spectres = d['noisy_spectres']
        self.noiseless_spectres = d['noiseless_spectres']
        self.lower_bound = d['bounds']['lower_bound']
        self.upper_bound = d['bounds']['upper_bound']
        self.background = d['floats']['background'][0]
        self.seed = d['floats']['seed'][0]
        self.n = d['floats']['n'][0]
        if d['kernel'].shape == () :
            self.p = Pinem(x=d['x'],
                           amplitude=d['floats']['amplitude'][0],
                           kernel=d['kernel']['kernel'].item(),
                           n_cutoff=d['floats']['n_cutoff'][0]
                        )
        else :
            self.p = Pinem(x=d['x'],
                           amplitude=d['floats']['amplitude'][0],
                           kernel=d['kernel'],
                           n_cutoff=d['floats']['n_cutoff'][0])
        
    
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

class TestPinem() :
    """
    Class to generate a sequence of controlled PINEM spectra (i.e. with known parameters) for testing purposes.

    TODO : Merge with DatasetPinem

    Args:
    - dataset_file (str): The path to the file to load the dataset from
    - params_dict (dict): A dictionary of lists with the parameters to vary. The keys are the names of the parameters and the values are the values to test.
    - n (int): The number of spectra to generate per parameter. The total number of spectra will be n**len(params_dict)
    - true_coords_keys (list[str]): The keys of the parameters to use as the true coordinates of the spectra. They should correspond to the output of the neural network.
    """
    def __init__(self, dataset_file : str, params_dict : dict, n : int = 20, true_coords_keys : list[str] = None) : 
        self.dataset_file = dataset_file
        self.params_dict = params_dict
        self.dataset = np.load(dataset_file)
        self.x = self.dataset['x']
        self.amplitude = self.dataset['floats']['amplitude'][0]
        kt = self.dataset['kernel']
        if not(kt.shape == ()):
            self.kernel = kt
        else : 
            self.kernel = kt['kernel'].item()
        self.n_cutoff = self.dataset['floats']['n_cutoff'][0]
        self.background = self.dataset['floats']['background'][0]
        self.n = n
        self.p = Pinem(x=self.x, amplitude=self.amplitude, kernel=self.kernel, n_cutoff=self.n_cutoff)
        self.true_coords_keys = true_coords_keys

    def generate_data(self) :
        """
        Generate the spectra according to the parameters in params_dict. It goes through all the combinations of the parameters and generates a spectrum for each combination.
        """
        truth = []
        data = [] 
        for params in tqdm(iterate_combinations(self.params_dict)):
            true_coords = [params[i] for i in self.params_dict.keys() if i in self.true_coords_keys]
            truth.append(true_coords)
            test = self.p.calc_sq_modulus(**params)
            test = np.random.poisson(test+ self.background*self.amplitude)
            data.append(test)

        self.data = np.array(data)
        self.truth = np.array(truth)
    
    def save_data(self, save_folder : str) :
        """
        Save the generated data to a file.

        Args:
        - save_folder (str): The path to the file to save the data to
        """
        np.savez(save_folder, data=self.data, truth=self.truth)
        print('Data saved in {}'.format(save_folder))



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

