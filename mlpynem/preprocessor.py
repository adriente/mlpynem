from sklearn.model_selection import train_test_split
import numpy as np
from scipy.ndimage import center_of_mass as com

class DataPreprocessor() :
    """
    Class to preprocess the data before training the neural network.
    It is used to first normalize the data and then split it into a training, validation and testing dataset.

    Args:
    - data (np.ndarray): The data to preprocess, i.e. all the spectra of the dataset object.
    - truth (np.ndarray): The true coordinates of the spectra, i.e. all the ground truth from the dataset object.
    - convolution (bool): Whether the data is to be used for a convolutional neural network or a dense neural network
    - normalization (str): The normalization method to use. It can be 'normalize', 'scale' or 'both'. See the normalize_data, scale_data methods for more information.
    - train_ratio (float): The ratio of the data to use for the training dataset
    - val_ratio (float): The ratio of the data to use for the validation dataset
    - test_ratio (float): The ratio of the data to use for the testing dataset (internal testing of keras)
    """
    def __init__(self, data, truth, convolution = False, normalization = None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, rem_bckgd = True) : 
        self.data = data
        self.truth = truth
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.normalization = normalization
        self.convolution = convolution
        self.rem_bckgd = rem_bckgd

    def eval_background(self, data) :
        """
        Evaluate the background of a set of spectra. The spectra should have the shape (number of spectra, spectra length).

        Args:
        - data (np.ndarray): The set of spectra to evaluate the background of.
        """
        # So far we take the first and last 10% of the spectra to evaluate the background
        return np.concatenate((data[:,:int(data.shape[1] * 0.1)], data[:,int(data.shape[1] * 0.9):]), axis=1).mean(axis=1).clip(min=0)
    
    def normalize_data(self, data) :
        """
        Normalize a set of spectra to the range [0, 1]. The spectra should have the shape (number of spectra, spectra length).

        Args:
        - data (np.ndarray): The set of spectra to normalize.
        """
        m = data.min(axis=1)[:,np.newaxis]
        M = data.max(axis=1)[:,np.newaxis]
        return (data - m) / (M - m)
    
    def scale_data(self, data) :
        """
        Scale a set of spectra to have a mean of 0 and a standard deviation of 1. The spectra should have the shape (number of spectra, spectra length).

        Args:
        - data (np.ndarray): The set of spectra to scale.
        """
        m = data.mean(axis=1)[:,np.newaxis]
        s = data.std(axis=1)[:,np.newaxis]
        return (data - m) / s
    
    def both(self, data) : 
        """
        First normalize and then scale a set of spectra. The spectra should have the shape (number of spectra, spectra length).

        Args:
        - data (np.ndarray): The set of spectra to normalize and scale.
        """
        return self.scale_data(self.normalize_data(data))

    def preprocess_and_split_data(self,data, truth,train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) : 
        """
        Preprocess the data and split it into a training, validation and testing dataset. It is mainly a wrapper around the train_test_split function from scikit-learn.

        Args:
        - data (np.ndarray): The data to preprocess, i.e. all the spectra of the dataset object.
        - truth (np.ndarray): The true coordinates of the spectra, i.e. all the ground truth from the dataset object.
        - train_ratio (float): The ratio of the data to use for the training dataset
        - val_ratio (float): The ratio of the data to use for the validation dataset
        - test_ratio (float): The ratio of the data to use for the testing dataset (internal testing of keras)
        """

        X_train, X_temp, y_train, y_temp = train_test_split(data, truth, train_size=train_ratio, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test) 

    def preprocess(self) :
        """
        Actual function to be called to preprocess the data and split it into a training, validation and testing dataset.

        Returns:
        - (X_train, y_train): The training dataset
        - (X_val, y_val): The validation dataset
        - (X_test, y_test): The testing dataset
        All those quantities can be directly fed to the neural network for training.
        """
        # First we remove the background
        if self.rem_bckgd :
            self.data = self.data - self.eval_background(self.data)[:,np.newaxis]

        self.data = correct_center_of_mass(self.data)
        if self.normalization is None : 
            # Note that this should not be used as neural networks are sensitive to the scale of the data
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.preprocess_and_split_data(self.data,
                                                                                                                          self.truth,
                                                                                                                          self.train_ratio,
                                                                                                                          self.val_ratio,
                                                                                                                          self.test_ratio) 
        elif self.normalization == 'normalize' : 
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.preprocess_and_split_data(self.normalize_data(self.data),
                                                                                                                          self.truth,
                                                                                                                          self.train_ratio,
                                                                                                                          self.val_ratio,
                                                                                                                          self.test_ratio)
        elif self.normalization == 'scale' :
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.preprocess_and_split_data(self.scale_data(self.data),
                                                                                                                          self.truth,
                                                                                                                          self.train_ratio,
                                                                                                                          self.val_ratio,
                                                                                                                          self.test_ratio)
        elif self.normalization == 'both' :
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.preprocess_and_split_data(self.both(self.data),
                                                                                                                          self.truth,
                                                                                                                          self.train_ratio,
                                                                                                                          self.val_ratio,
                                                                                                                          self.test_ratio)
            
        if self.convolution :
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
def center_of_mass(data : np.ndarray) :
    """
    Compute the center of mass of a collection of 1D array

    Parameters
    ----------
    data : 2D array
        The data to compute the center of mass with shape (number of spectra, number of points in the spectra)
    """ 
    coords = np.arange(data.shape[1])
    com = np.sum(data*coords, axis = 1)/np.sum(data, axis = 1)
    return com


def eval_center_of_mass(data) :
    """
    Evaluate the center of mass of a set of spectra. The spectra should have the shape (number of spectra, spectra length).

    Args:
    - data (np.ndarray): The set of spectra to evaluate the center of mass of.
    """
    cdata = data.copy()
    hM = cdata.max(axis = 1)/1.3
    mask = cdata < hM[:,np.newaxis]
    cdata[mask] = 0
    coms = center_of_mass(cdata)
    return coms
    
def correct_center_of_mass(data) :
    """
    Correct the center of mass of a set of spectra. The spectra should have the shape (number of spectra, spectra length).

    Args:
    - data (np.ndarray): The set of spectra to correct the center of mass of.
    """
    coms = eval_center_of_mass(data)
    for i in range(data.shape[0]) :
        data[i] = np.roll(data[i], int(np.round(data.shape[1]/2 - coms[i])))
    return data