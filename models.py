
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    quadratic = tf.minimum(tf.abs(error), delta)
    linear = tf.abs(error) - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(loss)

def custom_huber_metric(delta=1.0):
    def huber_metric(y_true, y_pred):
        error = y_true - y_pred
        quadratic = tf.minimum(tf.abs(error), delta)
        linear = tf.abs(error) - quadratic
        metric = 0.5 * quadratic**2 + delta * linear
        return tf.reduce_mean(metric)
    return huber_metric

class DataPreprocessor() :
    def __init__(self, data, truth, convolution = False, normalization = None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) : 
        self.data = data
        self.truth = truth
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.normalization = normalization
        self.convolution = convolution

    def normalize_data(self, data) :
        m = data.min(axis=1)[:,np.newaxis]
        M = data.max(axis=1)[:,np.newaxis]
        return (data - m) / (M - m)
    
    def scale_data(self, data) :
        m = data.mean(axis=1)[:,np.newaxis]
        s = data.std(axis=1)[:,np.newaxis]
        return (data - m) / s
    
    def both(self, data) : 
        return self.scale_data(self.normalize_data(data))

    def preprocess_and_split_data(self,data, truth,train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) : 

        X_train, X_temp, y_train, y_temp = train_test_split(data, truth, train_size=train_ratio, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test) 

    def preprocess(self) :
        if self.normalization is None : 
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
            
class BaseModel(ABC) :
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
        pass

    def train(self, X_train, y_train, X_val, y_val, epochs : int, batch_size : int) :
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), use_multiprocessing=True)
        return self.history

    def predict(self, X_test) : 
        return self.model.predict(X_test)
    
    def save_model(self, path : str) : 
        self.model.save(path)
        
    @property
    def weights(self) : 
        return self.model.get_weights()
    
    @weights.setter
    def weights(self, weights) : 
        self.model.set_weights(weights)

    def plot_history(self) : 
        if self.history : 
            plt.plot(self.history.history['loss'], label='train')
            plt.plot(self.history.history['val_loss'], label='val')
            plt.legend()
            plt.show()
        else : 
            raise AttributeError("Model has not been trained yet")


class DNNModel(BaseModel) : 
    def __init__(self, input_shape : int, num_params : int, mode : str, layers : list[int], dropouts : list[float], lr : float = 0.001, filename : str = None) :
        self.mode = mode
        self.layers = layers
        self.dropouts = dropouts
        self.lr = lr
        super().__init__(input_shape, num_params, filename)

    def make_model(self) : 
        activation_func = keras.layers.LeakyReLU(alpha=0.3)
    
        model = keras.Sequential()

        for i, layer in enumerate(self.layers) : 
            model.add(keras.layers.Dense(layer, activation=activation_func, input_shape=(self.input_shape,)))
            model.add(keras.layers.Dropout(self.dropouts[i]))
            
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
    


class CNNModel(BaseModel) : 
    def __init__(self, input_shape : int, num_params : int, mode : str, filters : list[int], kernel_size : list[int], dropouts : list[float], lr = 0.001, filename : str = None) :
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropouts = dropouts
        self.mode = mode
        self.lr = lr
        super().__init__(input_shape, num_params, filename)

    def make_model(self) :
        n_timesteps = self.input_shape
        n_features  = 1
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(n_timesteps,n_features)))
        for i, filter in enumerate(self.filters) : 
            model.add(keras.layers.Conv1D(filters=filter, kernel_size=self.kernel_size[i], activation='relu', input_shape=(n_timesteps,n_features)))
            model.add(keras.layers.Dropout(self.dropouts[i]))
        

        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(32, activation='relu'))
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

    
