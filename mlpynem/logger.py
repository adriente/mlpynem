import json

# TODO : I think this logging could be integrated in the neural network classes. It could also be improved to gather more informations.

class ModelInfo:
    """
    Class to store model information in json files. It will keep track of datasets, neural network models, and their results.
    Several iteration on a single neural network model can be stored in the same file. We keep track of the different trainings with the itreration number.

    Args:
    - filename (str): The filename of the json file to store the model information
    """
    def __init__(self, filename):
        self.filename = filename
        self.iteration = 0
        self.current_dict = {}
        

    def add_iteration(self, dataset_name, test_model_dataset_name):
        """
        Add a new iteration to the json file to store the infos of a new training of the neural network.

        Args:
        - dataset_name (str): The name of the dataset used for the training
        - test_model_dataset_name (str): The name of the dataset used for the testing of the model
        """
        self.iteration += 1
        self.current_dict['iteration_{}'.format(self.iteration)] = {'train_data' : dataset_name, 'test_data' : test_model_dataset_name}


    def save_model(self,model):
        """
        Save the model information in the json file, i.e. layers, kernel sizes, etc.

        Args:
        - model (keras.models.Model): The neural network model to save
        """
        init_dict = model.__dict__.copy()
        try :
            del init_dict['filename']
            del init_dict['model']
        except KeyError:
            pass

        self.current_dict['iteration_{}'.format(self.iteration)]['model'] = init_dict
    
    def save_history(self, model) : 
        """
        Save the history of the training of the model in the json file. Currently, it only saves the last epoch of the training.

        Args:
        - model (keras.models.Model): The neural network model to save the history of
        """
        ends = {}
        for key in model.history.history.keys() : 
            ends[key] = model.history.history[key][-1]
        self.current_dict['iteration_{}'.format(self.iteration)]['history'] = ends

    def save_norm(self,normalization) : 
        """
        Saves the normalization method of the dataset used for the training of the model.

        Args:
        - normalization (str): The normalization method used for the training dataset
        """
        self.current_dict['iteration_{}'.format(self.iteration)]['normalization'] = normalization

    def save_test_results(self, results) :
        """
        Saves the results of the testing of the model on the test dataset.
        It saves the mean and standard deviation on the error between the true coordinates and the predicted coordinates.

        Args:
        - results (dict): The results of the testing of the model on the test dataset
        """
        self.current_dict['iteration_{}'.format(self.iteration)]['test_results'] = results   

    def save(self): 
        """
        Save the current state of the ModelInfo object to the json file.
        To be called at the end of the training and testing of the model as it will overwrite the file.
        """
        with open(self.filename, 'w') as f:
            json.dump(self.current_dict, f, indent=4)