from pinem import Pinem
import numpy as np
from tqdm import tqdm
import itertools

# def gaussian(x, cen, sig):
#     return np.exp(-(x-cen)**2/(2*sig**2))

class DatasetPinem():
    def __init__(
        self,
        x: np.ndarray,
        amplitude: np.ndarray,
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
            
            # self.espm_fit()
            
        else : 
            self.load_model(load)


    def expand_bounds(self) : 
        lb = np.expand_dims(self.lower_bound, axis=1)
        ub = np.expand_dims(self.upper_bound, axis=1)
        nlb = lb@(np.ones((1,self.n)))
        nub = ub@(np.ones((1,self.n)))
        return nlb, nub
    
    # For now, only works for gaussian kernel, omega fixed and fwhm fixed
    # def gen_g_matr(self, omega,fwhm) : 
    #     cens = [i*omega for i in range(-50,50) if ((i*omega) > self.p.x[0]) & ((i*omega) < self.p.x[-1])]
    #     self.g_matr = np.zeros((self.p.x.shape[0],len(cens)))
    #     for i,cen in enumerate(cens) : 
    #         self.g_matr[:,i] = gaussian(self.p.x, cen = cen, sig = fwhm/2.355)
        #self.estim = SmoothNMF(n_components = 1, force_simplex=False, G = self.g_matr)

    def gen_data(self) :
        np.random.seed(self.seed)
        elb, eub = self.expand_bounds()
        self.xs = (np.random.uniform(elb, eub, size=(5,self.n))).T
        len_x = self.p.x.shape[0]
        self.spectres = np.zeros((self.n, len_x))
        # self.gen_g_matr(self.xs[0][0], self.xs[0][4])
        # g rt fwhm omega offset
        for i, x in tqdm(enumerate(self.xs)) :
            self.spectres[i,:] = (self.p.calc_sq_modulus(omega=x[3], g =  x[0], offset=  x[4], fwhm = x[2], rt = x[1]))[:,np.newaxis].T   
        self.noiseless_spectres = self.spectres + self.amplitude*self.background

        self.noisy_spectres = np.random.poisson(self.spectres+self.background*self.amplitude)
        #self.noiseless_spectres = self.spectres*self.amplitude
        

    # def espm_fit(self) :
    #     self.weigths = np.zeros((self.n, self.g_matr.shape[1])) 
    #     if self.noisy :
    #         compute = self.noisy_spectres
    #     else :
    #         compute = self.spectres
    #     for i, sp in enumerate(compute) : 
    #         estim = SmoothNMF(n_components = 1, force_simplex=False, G = self.g_matr, init = 'nndsvd')
    #         estim.fit(sp[:,np.newaxis])
    #         self.weigths[i,:] = normalize_W2( estim.W_[:,0])
            

    def __len__(self):
        return self.n

    # def __getitem__(self, idx): 
    #     #window = blackman(len_x) 
    #     spectre = self.input_spectres[idx][:, np.newaxis]
    #     output = self.xs[idx][:2]
    #     # if self.noisy:
    #     #     interm = np.random.poisson(spectre)
    #     #     #fft = np.abs(rfft(interm*window))#[0:len_x//2]
    #         #fft = (fft - np.min(fft)) / (np.max(fft) - np.min(fft))
    #         #input = np.hstack((interm, fft))[:, np.newaxis]
    #         # estim = SmoothNMF(n_components = 1, force_simplex=False, G = self.g_matr, init = 'nndsvd')
    #         # estim.fit(interm)
    #         # input = normalize_W(estim.W_)

    #     # else:
    #     #     interm = spectre
    #     #     # fft = np.abs(rfft(interm*window))#[0:len_x//2]
    #     #     # fft = (fft - np.min(fft)) / (np.max(fft) - np.min(fft))
    #     #     # input = np.hstack((interm, fft))[:, np.newaxis]
    #     #     estim = SmoothNMF(n_components = 1, force_simplex=False, G = self.g_matr, init = 'nndsvd')
    #     #     estim.fit(interm)
    #     #     input = normalize_W(estim.W_)
    #     return (torch.tensor(spectre, dtype=torch.float32)).T, torch.tensor(
    #         output, dtype=torch.float32
    #     )
    
    def save_model(self, path) :
        d = {}
        float_dt = np.dtype([('amplitude', np.float64),
                             ('n_cutoff', np.int64),
                            #  ('k_cutoff', np.int64),
                             ('n', np.int64),
                             ('background', np.float64),
                             ('seed', np.int64)])
        float_arr = np.array([(self.p.amplitude, self.p.n_cutoff, self.n, self.background, self.seed)], dtype=float_dt)
        bounds_dt = ['lower_bound', 'upper_bound']
        bounds_arr = np.rec.fromarrays([self.lower_bound, self.upper_bound], names=bounds_dt)
        d['x'] = self.p.x
        # d['g_matr'] = self.g_matr
        # d['weights'] = self.weigths
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
        d = np.load(path, allow_pickle=True)
        self.xs = d['xs']
        self.spectres = d['spectres']
        self.noisy_spectres = d['noisy_spectres']
        self.noiseless_spectres = d['noiseless_spectres']
        # self.g_matr = d['g_matr']
        # self.weigths = d['weights']
        self.lower_bound = d['bounds']['lower_bound']
        self.upper_bound = d['bounds']['upper_bound']
        self.background = d['floats']['background'][0]
        self.seed = d['floats']['seed'][0]
        self.n = d['floats']['n'][0]
        if d['kernel'].shape == () :
            self.p = Pinem(x=d['x'],
                           amplitude=d['floats']['amplitude'][0],
                           kernel=d['kernel']['kernel'].item(),
                           n_cutoff=d['floats']['n_cutoff'][0],
                        )#    k_cutoff=d['floats']['k_cutoff'][0])
        else :
            self.p = Pinem(x=d['x'],
                           amplitude=d['floats']['amplitude'][0],
                           kernel=d['kernel'],
                           n_cutoff=d['floats']['n_cutoff'][0],
                           k_cutoff=d['floats']['k_cutoff'][0])
        
    
def normalize_spectra(spectra: np.ndarray) -> np.ndarray:
    """
    Normalize a set of spectra to the range [0, 1]
    """
    m = np.min(spectra, axis=1)
    M = np.max(spectra, axis=1)
    int_sp = (spectra - m[:, np.newaxis]) / (M - m)[:, np.newaxis]
    if np.any(np.isnan(int_sp)) : 
        print('nan')
        print(np.argwhere(np.isnan(int_sp)))
    return (spectra - m[:, np.newaxis]) / (M - m)[:, np.newaxis]

# def normalize_W(W) -> np.ndarray:
#     """
#     Normalize a set of spectra to the range [0, 1]
#     """
#     m = np.min(W, axis=0)
#     M = np.max(W, axis=0)
#     return (W - m[:, np.newaxis]) / (M - m)[:, np.newaxis]

# def normalize_W2(W) -> np.ndarray:
#     """
#     Normalize a set of spectra to the range [0, 1]
#     """
#     m = np.min(W, axis=0)
#     M = np.max(W, axis=0)
#     return (W - m) / (M - m)

class TestPinem() :
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
        np.savez(save_folder, data=self.data, truth=self.truth)
        print('Data saved in {}'.format(save_folder))



def iterate_combinations(params_dict):
    keys = params_dict.keys()
    values = params_dict.values()
    combinations = list(itertools.product(*values))
    for combination in combinations:
        yield dict(zip(keys, combination))

