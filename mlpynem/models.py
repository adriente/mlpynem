
import tensorflow as tf
from tensorflow import keras
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber loss function. It is a combination of the L1 and L2 loss functions.
    It is used to reduce the sensitivity of the model to outliers in the data.

    Args:
    - y_true (tf.Tensor): The true values of the data
    - y_pred (tf.Tensor): The predicted values of the data
    - delta (float): The threshold for the quadratic loss
    """
    error = y_true - y_pred
    if tf.abs(error) <= delta:
        return tf.reduce_mean(0.5 * error**2)
    else:
        return tf.reduce_mean(delta * (tf.abs(error) - 0.5 * delta))
    # TODO : Ask Mouloud as to why he implemented it this way
    # quadratic = tf.minimum(tf.abs(error), delta)
    # linear = tf.abs(error) - quadratic
    # loss = 0.5 * quadratic**2 + delta * linear
    # return tf.reduce_mean(loss)

def custom_huber_metric(delta=1.0):
    """
    Huber metric function. It is a combination of the L1 and L2 loss functions.
    It is used to reduce the sensitivity of the model to outliers in the data.

    Args:
    - delta (float): The threshold for the quadratic loss
    """
    def huber_metric(y_true, y_pred):
        return huber_loss(y_true, y_pred, delta=delta)
    # TODO : Ask Mouloud as to why he implemented it this way
    # def huber_metric(y_true, y_pred):
    #     error = y_true - y_pred
    #     quadratic = tf.minimum(tf.abs(error), delta)
    #     linear = tf.abs(error) - quadratic
    #     metric = 0.5 * quadratic**2 + delta * linear
    #     return tf.reduce_mean(metric)
    return huber_metric
            
class BaseModel(ABC) :
    """
    Abstract base class for the neural network models. It is used to define the common methods and attributes of the neural network models.

    Args:
    - input_shape (int): The shape of the input data, i.e. the number of features of the data. For a 1D dataset, it is the length of the spectra.
    - num_params (int): The number of parameters to predict. It should have the same length as the ground truth of the dataset.
    - filename (str): The path to the file to load the model from. If None, the model will be created from scratch.
    """
    def __init__(self, input_shape : int, num_params : int, filename : str = None) : 
        self.input_shape = input_shape
        self.num_params = num_params
        self.filename = filename
        if filename : 
            self.model = keras.models.load_model(filename)
        else :
            self.model = self.make_model()

    @abstractmethod
    def make_model(self) :
        """
        Abstract method to create the neural network model. It should be implemented in the child classes.
        """ 
        pass

    def train(self, X_train, y_train, X_val, y_val, epochs : int, batch_size : int) :
        """
        Train the neural network model on the training dataset. It is a wrapper around the fit method of the keras.models.Model class.

        Args:
        - X_train (np.ndarray): The training dataset
        - y_train (np.ndarray): The true coordinates of the training dataset
        - X_val (np.ndarray): The validation dataset
        - y_val (np.ndarray): The true coordinates of the validation dataset
        - epochs (int): The number of epochs to train the model
        - batch_size (int): The batch size to use for the training
        """
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), use_multiprocessing=True)
        return self.history

    def predict(self, X_test) : 
        """
        Make a prediction on a set of data. It is a wrapper around the predict method of the keras.models.Model class.

        Args:
        - X_test (np.ndarray): The data to make a prediction on. It should have a shape of (number of spectra, spectra length)

        Returns:
        - np.ndarray: The predicted coordinates of the data with shape (number of spectra, number of parameters)
        """
        return self.model.predict(X_test)
    
    def save_model(self, path : str) :
        """
        Save the model to a hdf5 file. It is a wrapper around the save method of the keras.models.Model class.
        
        Args:
        - path (str): The path to save the model to
        """ 
        self.model.save(path)
        
    @property
    def weights(self) : 
        """
        Get the weights of the model. It is a wrapper around the get_weights method of the keras.models.Model class.
        This function helps you get the weights from a previous training.
        """
        return self.model.get_weights()
    
    @weights.setter
    def weights(self, weights) : 
        """
        Set the weights of the model. It is a wrapper around the set_weights method of the keras.models.Model class.
        This function helps you set the weights from a previous training to a new neural network.
        """
        self.model.set_weights(weights)

    def plot_history(self) : 
        """
        Plot the history of the training of the model. It shows you the training and validation loss to check for convergence.
        """
        if self.history : 
            plt.plot(self.history.history['loss'], label='train')
            plt.plot(self.history.history['val_loss'], label='val')
            plt.legend()
            plt.show()
        else : 
            raise AttributeError("Model has not been trained yet")


class DNNModel(BaseModel) :
    """
    Class to create a dense neural network model. It is a child class of the BaseModel class.

    Args:
    - input_shape (int): The shape of the input data, i.e. the number of features of the data. For a 1D dataset, it is the length of the spectra.
    - num_params (int): The number of parameters to predict. It should have the same length as the ground truth of the dataset.
    - mode (str): The mode of the neural network. It can be 'poisson' or 'normal'. It is used to define the loss function of the neural network.
    - layers (list[int]): The number of neurons in each layer of the neural network
    - dropouts (list[float]): The dropout rate to use for each layer of the neural network
    - lr (float): The learning rate of the neural network
    - filename (str): The path to the file to load the model from. If None, the model will be created from scratch.
    """ 
    def __init__(self, input_shape : int, num_params : int, mode : str, layers : list[int], dropouts : list[float], lr : float = 0.001, filename : str = None) :
        self.mode = mode
        self.layers = layers
        self.dropouts = dropouts
        self.lr = lr
        super().__init__(input_shape, num_params, filename)

    def make_model(self) :
        """
        Builds the neural network model. It is a dense neural network with leaky ReLU activation functions and dropout layers.
        There len(self.layers)+1 layers in the neural network. The first len(self.layers) layers have a leaky ReLU activation function and a dropout layer.
        The last layer has a linear activation function.
        """ 
        activation_func = keras.layers.ReLU()
    
        model = keras.Sequential()

        for i, layer in enumerate(self.layers) : 
            model.add(keras.layers.Dense(layer, activation=activation_func, input_shape=(self.input_shape,)))
            model.add(keras.layers.Dropout(self.dropouts[i]))
            
        model.add(keras.layers.Dense(self.num_params, activation='linear'))

        if self.mode == 'poisson': 
            # Technically it is not a poisson loss function but a huber loss function. To be modified if necessary.      
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), 
                        loss=huber_loss,
                        metrics=["mse", "mae", custom_huber_metric(1)]
                        )
        
        elif self.mode == "normal":
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                        loss="mse",
                        metrics=["mse","mae"]
                        )
            
        return model
    


class CNNModel(BaseModel) : 
    """
    Class to create a convolutional neural network model. It is a child class of the BaseModel class.

    Args:
    - input_shape (int): The shape of the input data, i.e. the number of features of the data. For a 1D dataset, it is the length of the spectra.
    - num_params (int): The number of parameters to predict. It should have the same length as the ground truth of the dataset.
    - mode (str): The mode of the neural network. It can be 'poisson' or 'normal'. It is used to define the loss function of the neural network.
    - filters (list[int]): The number of filters in each layer of the neural network
    - kernel_size (list[int]): The size of the kernel in each layer of the neural network
    - dropouts (list[float]): The dropout rate to use for each layer of the neural network
    - lr (float): The learning rate of the neural network
    - filename (str): The path to the file to load the model from. If None, the model will be created from scratch.
    """
    def __init__(self, input_shape : int, num_params : int, mode : str, filters : list[int], kernel_size : list[int], dropouts : list[float], lr = 0.001, filename : str = None) :
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropouts = dropouts
        self.mode = mode
        self.lr = lr
        super().__init__(input_shape, num_params, filename)

    def make_model(self) :
        """
        Builds the neural network model. It is a convolutional neural network with ReLU activation functions and dropout layers.
        The first len(self.filters) layers have a convolutional layer with a ReLU activation function and a dropout layer.
        The last layers consist of a max pooling layer, a flatten layer, and two dense layers with ReLU  and linear activation functions.
        """
        n_timesteps = self.input_shape
        n_features  = 1
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(n_timesteps,n_features)))
        for i, filter in enumerate(self.filters) : 
            model.add(keras.layers.Conv1D(filters=filter, kernel_size=self.kernel_size[i], strides = 2, activation='relu', input_shape=(n_timesteps,n_features)))
            #model.add(keras.layers.Dropout(self.dropouts[i]))
        
        # For now this block is something I copied from the internet : https://colab.research.google.com/drive/1zjh0tUPYJYgJJunpLC9fW5uf--O0LKeZ?usp=sharing. I don't know if it is the best way to do it.
        #model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Flatten())
        #model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(self.num_params, activation='linear'))

        if self.mode == 'poisson':       
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), 
                        loss=huber_loss,
                        metrics=["mse", "mae", custom_huber_metric(1)]
                        )
        
        elif self.mode == "normal":
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                        loss="mse",
                        metrics=["mse","mae"]
                        )
        return model
    
# TODO : improve the control on the neural networks. Improve flexibility of the structure of the neural networks.

    
