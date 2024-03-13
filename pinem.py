# import numpy as np
# from scipy.special import jv


# def gaussian(x, mu, sig):
#     return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


# class Pinem:
#     def __init__(self, x, amplitude, width, offset, background, pump):
#         self.x = x
#         self.amplitude = amplitude
#         self.width = width
#         self.offset = offset
#         self.background = background
#         self.pump = pump

#     def calc(self, g_map):
#         f = self.background
#         photon = 1240 / self.pump
#         vec_g = g_map.reshape(-1)
#         for n in range(-10, 10):
#             jv_map = jv(n, 2 * vec_g) ** 2
#             f += (
#                 self.amplitude
#                 * jv_map
#                 * gaussian(self.x, self.offset - n * photon, self.width)[:, None]
#             )
#         return np.float64(f).reshape((self.x.shape[0], g_map.shape[0], g_map.shape[1]))

#     def calc_vec(self, g_vec):
#         f = self.background
#         photon = 1240 / self.pump
#         for n in range(-10, 10):
#             jv_map = jv(n, 2 * g_vec) ** 2
#             f += (
#                 self.amplitude
#                 * jv_map
#                 * gaussian(self.x, self.offset - n * photon, self.width)[:, None]
#             )
#         return np.float64(f).reshape((self.x.shape[0], g_vec.shape[0]))
    

# import numpy as np
# from scipy.special import jv


# def gaussian(x, mu, sig):
#     return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


# class Pinem:
#     def __init__(self, x, kernel, amplitude, pump, k_cutoff, n_cutoff):
#         """
#         x: 1D array of energy values in eV
#         kernel: 1D array of the zero-loss peak in the same energy units as x and same length as x
#         The kernel is assumed to be centered at x=0
#         amplitude: float to scale the intensity of the peaks
#         pump: float describing the energy of the subbands in eV
#         k_cutoff: int describing the number of subbands to consider
#         n_cutoff: int describing the number of subbands to consider
#         """
#         self.x = x
#         self.amplitude = amplitude
#         self.kernel = kernel
#         self.k_cutoff = k_cutoff
#         self.n_cutoff = n_cutoff
#         self.pump = pump
#         self.scale = self.x[1] - self.x[0]
#         self.ns, self.ks, self.mns, self.mks = self.gen_indices()
#         self.kernel_matrix, self.mask_n = self.gen_kernel_mattrix()

#     def gen_indices(self):
#         ns = np.arange(-self.n_cutoff, self.n_cutoff+1)
#         ks = np.arange(-self.k_cutoff, self.k_cutoff+1)
#         mns, mks = np.meshgrid(ns, ks)
#         return ns, ks, mns, mks

#     def gen_kernel_mattrix(self):
#         mus = (self.ns*self.pump/self.scale).astype(int)
#         mask_mus = np.abs(mus) < self.x.shape[0]/2
#         len_mask = np.sum(mask_mus)
#         kernels = np.repeat(self.kernel[:,np.newaxis], len_mask, axis=1)
#         t = np.array([np.roll(kernels[:,i], mus[mask_mus][i]) for i in range(len(mus[mask_mus]))])
#         return t, mask_mus

#     def calc(self, g1, g2, theta):
#         j2 = (jv(self.ks, 2*g2)*np.exp(1j*self.ks*theta))[:,np.newaxis]
#         j1 = jv(self.mns-2*self.mks, 2*g1)
#         js = np.sum(j2*j1, axis=0)[self.mask_n][:,np.newaxis]
#         wave = self.amplitude*np.sum(self.kernel_matrix*js, axis=0)

#         return wave
    
#     def calc_sq_modulus(self, g1, g2, theta):
#         wave = self.calc(g1, g2, theta)
#         return np.real(wave)**2 + np.imag(wave)**2
    

# def bessel_mat_product(x,g1, g2, theta, n_cutoff, k_cutoff, omega, fwhm) : 
#     ks = np.arange(-k_cutoff, k_cutoff+1)
#     ns = np.arange(-n_cutoff, n_cutoff+1)
#     mns, mks = np.meshgrid(ns, ks)
#     j2 = (jv(ks, 2*g2)*np.exp(1j*ks*theta))[:,np.newaxis]
#     j1 = jv(mns-2*mks, 2*g1)
#     js = np.sum(j2*j1, axis=0)[:,np.newaxis]
#     mus = ns*omega

#     x_matrix, mu_matrix = np.meshgrid(x,mus)
#     wave = np.sum(gaussian(x_matrix, mu_matrix, fwhm=fwhm)*js, axis=0)

#     return wave

import numpy as np
from scipy.special import jv, voigt_profile

# We use this implementation so that the shape is not always the same at the cost of some accuracy on fwhm
def voigt_fwhm(x, fwhm, seed = 0) : 
    if seed == 0:
        seed = np.random.randint(100000)
    np.random.seed(seed)
    ratio = 3.0*np.random.rand(1)
    gamma  = fwhm/(0.5346 + np.sqrt(0.2166 + ratio**2)) #inverting the approximation of https://doi.org/10.1364/JOSA.63.000987
    # it leads to roughly 10% error in the fwhm
    sigma = gamma*ratio
    profile = voigt_profile(x, sigma, gamma)
    return profile/np.max(profile)

def my_gaussian(x, mu, sig):
    gauss = np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))*(1/(sig*np.sqrt(2*np.pi)))
    return gauss


def g_distrib_temp_averaged(g,g0,ratio):
    return ((1./np.sqrt(np.pi)*ratio)*(g/g0)**(ratio*ratio))*1./(g*np.sqrt(np.log(g0/g)))


class Pinem:
    def __init__(self, x, amplitude,n_cutoff, kernel = None):
        """
        x: 1D array of energy values in eV
        kernel: 1D array of the zero-loss peak in the same energy units as x and same length as x
        The kernel is assumed to be centered at x=0
        amplitude: float to scale the intensity of the peaks
        pump: float describing the energy of the subbands in eV
        k_cutoff: int describing the number of subbands to consider
        n_cutoff: int describing the number of subbands to consider
        """
        self.x = x
        self.delta_g = np.linspace(0.0001,1.0-0.0001,1000)
        self.num_g = self.delta_g.shape[0]
        self.amplitude = amplitude
        self.kernel = kernel
        self.n_cutoff = n_cutoff
        self.scale = self.x[1] - self.x[0]
        self.ns, self.fns = self.gen_indices()


    def gen_indices(self):
        tns = np.arange(-self.n_cutoff, self.n_cutoff+1)
        ns = tns[:,np.newaxis]*np.ones((self.num_g,))
        return ns, tns

    def gen_kernel_mattrix(self, omega, offset = 0.0,fwhm= 0.3, seed = 0):
        # To be investigated but I don't think it should change much to use ks or ns.
        fmus = self.fns*omega/self.scale + offset/self.scale
        mus = np.round(fmus).astype(int)
        
        mask_mus = np.abs(mus) < self.x.shape[0]/2
        len_mask = np.sum(mask_mus)
        if self.kernel == 'Voigt' : 
            kernel1D = voigt_fwhm(self.x, fwhm, seed = seed)
        elif self.kernel == 'Gaussian' : 
            kernel1D = my_gaussian(self.x, 0, fwhm/2.355)
        else :
            kernel1D = np.roll(self.kernel)
        kernels = np.repeat(kernel1D[:,np.newaxis], len_mask, axis=1)
        t = np.array([np.roll(kernels[:,i], mus[mask_mus][i]) for i in range(len(mus[mask_mus]))])
        return t, mask_mus

    def calc(self, omega,g,offset=0.0,fwhm = 0.3, rt = 0.7, seed = 0):
        # mod = np.abs(omega2/omega1 - round_ratio)
        # if mod > self.interference_cutoff:
        #     n_kern_mat, mask_n = self.gen_kernel_mattrix(omega1, ks = False, fwhm = fwhm, seed = seed)
        #     k_kern_mat, mask_k = self.gen_kernel_mattrix(omega2, ks = True, fwhm = fwhm, seed = seed)
        #     j2 = jv(self.ks, 2*g2)[:,np.newaxis][mask_k]*k_kern_mat
        #     j1 = jv(self.ns, 2*g1)[:,np.newaxis][mask_n]*n_kern_mat
        #     wave =self.amplitude*( j2.sum(axis = 0) + j1.sum(axis = 0))
        #     self.kernel_matrix = n_kern_mat
        # else : 
        vg1 = np.tile((g*self.delta_g)[:,np.newaxis],self.ns.shape[0]).T
        g_dist = g_distrib_temp_averaged(vg1, g, rt)
        assert np.isnan(g_dist).sum() == 0, 'Il y a un souci vg1 max {}, vg1 min {}, g {}'.format(np.max(vg1), np.min(vg1), g)
        kern_mat, mask_n = self.gen_kernel_mattrix(omega, offset=offset, fwhm = fwhm, seed = seed)
        j = jv(self.ns, 2*vg1)
        js = np.sum((g_dist*j**2)[mask_n,:],axis = 1)[:,np.newaxis]
        wave = np.sum(kern_mat*js, axis=0)# + np.random.normal(background*self.amplitude, background*self.amplitude, size = self.x.shape[0])).clip(min = 0.0)
        self.kernel_matrix = kern_mat

        return wave
    
    def calc_sq_modulus(self, omega, g, offset = 0.0,fwhm = 0.3, rt = 0.7, seed = 0):
        wave = self.calc(omega,g, offset = offset,fwhm = fwhm, rt = rt, seed = seed)
        mod_wave = np.real(wave)**2 + np.imag(wave)**2
        Mwave, mwave = np.max(mod_wave), np.min(mod_wave)
        nwave = (mod_wave - mwave)/(Mwave - mwave)
        fwave = self.amplitude*nwave
        return fwave
    
        # omgs = omega1* self.mns + omega2*self.mks
        # allowed = np.where(omgs - omgs.astype(int) ==0,self.mns - self.mks,np.inf)


# # We use this implementation so that the shape is not always the same at the cost of some accuracy on fwhm
# def voigt_fwhm(x, fwhm, seed = 0) : 
#     if seed == 0:
#         seed = np.random.randint(100000)
#     np.random.seed(seed)
#     ratio = 3.0*np.random.rand(1)
#     gamma  = fwhm/(0.5346 + np.sqrt(0.2166 + ratio**2)) #inverting the approximation of https://doi.org/10.1364/JOSA.63.000987
#     # it leads to roughly 10% error in the fwhm
#     sigma = gamma*ratio
#     profile = voigt_profile(x, sigma, gamma)
#     return profile/np.max(profile)

# def gaussian(x, mu, sig):
#     return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))*(1/(sig*np.sqrt(2*np.pi)))


# class Pinem:
#     def __init__(self, x, amplitude, k_cutoff, n_cutoff, kernel = None, interference_cutoff = 1e-1):
#         """
#         x: 1D array of energy values in eV
#         kernel: 1D array of the zero-loss peak in the same energy units as x and same length as x
#         The kernel is assumed to be centered at x=0
#         amplitude: float to scale the intensity of the peaks
#         pump: float describing the energy of the subbands in eV
#         k_cutoff: int describing the number of subbands to consider
#         n_cutoff: int describing the number of subbands to consider
#         """
#         self.x = x
#         self.amplitude = amplitude
#         self.kernel = kernel
#         self.k_cutoff = k_cutoff
#         self.n_cutoff = n_cutoff
#         self.scale = self.x[1] - self.x[0]
#         self.ns, self.ks, self.mns, self.mks = self.gen_indices()
#         self.interference_cutoff = interference_cutoff

#     def gen_indices(self):
#         ns = np.arange(-self.n_cutoff, self.n_cutoff+1)
#         ks = np.arange(-self.k_cutoff, self.k_cutoff+1)
#         mns, mks = np.meshgrid(ns, ks)
#         return ns, ks, mns, mks

#     def gen_kernel_mattrix(self, omega, ks = False ,fwhm= 0.3, seed = 0):
#         # To be investigated but I don't think it should change much to use ks or ns.
#         if ks :
#             mus = (self.ks*omega/self.scale).astype(int)
#         else : 
#             mus = (self.ns*omega/self.scale).astype(int)
        
#         mask_mus = np.abs(mus) < self.x.shape[0]/2
#         len_mask = np.sum(mask_mus)
#         if self.kernel == 'Voigt' : 
#             kernel1D = voigt_fwhm(self.x, fwhm, seed = seed)
#         elif self.kernel == 'Gaussian' : 
#             kernel1D = gaussian(self.x, 0, fwhm/2.355)
#         else :
#             kernel1D = self.kernel
#         kernels = np.repeat(kernel1D[:,np.newaxis], len_mask, axis=1)
#         t = np.array([np.roll(kernels[:,i], mus[mask_mus][i]) for i in range(len(mus[mask_mus]))])
#         return t, mask_mus

#     def calc(self, omega1, omega2 ,  g1, g2, theta,fwhm = 0.3, seed = 0):
#         round_ratio = np.round(omega2/omega1)
#         mod = np.abs(omega2/omega1 - round_ratio)
#         if mod > self.interference_cutoff:
#             n_kern_mat, mask_n = self.gen_kernel_mattrix(omega1, ks = False, fwhm = fwhm, seed = seed)
#             k_kern_mat, mask_k = self.gen_kernel_mattrix(omega2, ks = True, fwhm = fwhm, seed = seed)
#             j2 = jv(self.ks, 2*g2)[:,np.newaxis][mask_k]*k_kern_mat
#             j1 = jv(self.ns, 2*g1)[:,np.newaxis][mask_n]*n_kern_mat
#             wave =self.amplitude*( j2.sum(axis = 0) + j1.sum(axis = 0))
#             self.kernel_matrix = n_kern_mat
#         else : 
#             kern_mat, mask_n = self.gen_kernel_mattrix(omega1, ks = False, fwhm = fwhm, seed = seed)
#             j2 = (jv(self.ks, 2*g2)*np.exp(1j*self.ks*theta))[:,np.newaxis]
#             j1 = jv(self.mns-round_ratio*self.mks, 2*g1)
#             js = np.sum(j2*j1, axis=0)[mask_n][:,np.newaxis]
#             wave = self.amplitude*np.sum(kern_mat*js, axis=0)
#             self.kernel_matrix = kern_mat

#         return wave
    
#     def calc_sq_modulus(self, omega1, omega2 ,  g1, g2, theta,fwhm = 0.3, seed = 0):
#         wave = self.calc(omega1, omega2 ,  g1, g2, theta,fwhm = fwhm, seed = seed)
#         return np.real(wave)**2 + np.imag(wave)**2
    
#         # omgs = omega1* self.mns + omega2*self.mks
#         # allowed = np.where(omgs - omgs.astype(int) ==0,self.mns - self.mks,np.inf)
