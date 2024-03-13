import json

class ModelInfo:
    def __init__(self, filename):
        self.filename = filename
        self.iteration = 0
        self.current_dict = {}
        

    def add_iteration(self, dataset_name, test_model_dataset_name):
        self.iteration += 1
        self.current_dict['iteration_{}'.format(self.iteration)] = {'train_data' : dataset_name, 'test_data' : test_model_dataset_name}


    def save_model(self,model):
        init_dict = model.__dict__.copy()
        try :
            del init_dict['filename']
            del init_dict['model']
        except KeyError:
            pass

        self.current_dict['iteration_{}'.format(self.iteration)]['model'] = init_dict
    
    def save_history(self, model) : 
        ends = {}
        for key in model.history.history.keys() : 
            ends[key] = model.history.history[key][-1]
        self.current_dict['iteration_{}'.format(self.iteration)]['history'] = ends

    def save_norm(self,normalization) : 
        self.current_dict['iteration_{}'.format(self.iteration)]['normalization'] = normalization

    def save_test_results(self, results) :
        self.current_dict['iteration_{}'.format(self.iteration)]['test_results'] = results   

    def save(self): 
        with open(self.filename, 'w') as f:
            json.dump(self.current_dict, f, indent=4)