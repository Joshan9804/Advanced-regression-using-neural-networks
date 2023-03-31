import torch
import pickle
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from torch import nn
import torch
import traceback
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None
class Regressor():

    def __init__(self, x, neurons=[1], activations=["relu"], learning_rate = 0.001, batch_size = 64, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - neurons {list[int]} -- list specifying the number of
                neurons in each linear layer (length = depth of the nn)
            - activations {list[str]} -- list containing the activation
                functions after each linear layer
            - learning_rate {float} -- learning rate to train the network
            - nb_epoch {int} -- number of epochs to train the network.
            - batch_size {int} -- number of instances in each batch

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        
        # attributes related to preprocessing
        self.x = x
        self.median_train_dict=dict() # Stores all median values for training data.
        self.min_train = pd.DataFrame({}) # stores min values
        self.max_train = pd.DataFrame({}) # stores max train values
        self.one_hot_encoded_data_train_min=0
        self.one_hot_encoded_data_train_max=0
        self.cat_dummies=list() #stores all training dummy variables
        self.processed_columns=list() #stores all training columns including dummy variables

        # attributes related to the neural network model
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.neurons = neurons
        self.activations = activations
  
        # build layers specified by self.neurons and self.activations
        self._layers = []
        for layer_num in range(len(self.neurons)):
            # add linear layer
            n_out = self.neurons[layer_num]
            if layer_num == 0:
                n_in = self.input_size
            else:
                n_in = self.neurons[layer_num-1]
            self._layers.append(nn.Linear(n_in, n_out))
            
            # add activation function
            if self.activations[layer_num] == "relu":
                self._layers.append(nn.ReLU())
            else:
                self._layers.append(nn.Sigmoid())
            
        # build torch neural network with the specified layers
        self.model = nn.Sequential(*self._layers)
        print(f"model: {self.model}")
        
        # initialize weights randomly 
        self.model.apply(self._weights_init)
        
        # define loss and optimimser
        self.mse_loss = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(), self.learning_rate) 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    
    def _preprocessor(self, x, y = None, training = False):
            """ 
            Preprocess input of the network.

            Arguments:
                - x {pd.DataFrame} -- Raw input array of shape 
                    (batch_size, input_size).
                - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
                - training {boolean} -- Boolean indicating if we are training or 
                    testing the model.

            Returns:
                - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
                  size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
                - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
                  size (batch_size, 1).

            """
            #######################################################################
            #                       ** START OF YOUR CODE **
            #######################################################################

            # Replace this code with your own
            # Return preprocessed x and y, return None for y if it was None

            if training==True:
                # Replace all x Nan with median
                x_columns=x.select_dtypes(include=['float']).columns
                self.median_train_dict=dict()

                def replace_NA(df,column_name):
                    median_col=df[column_name].median()
                    self.median_train_dict[column_name]=median_col
                    df[column_name].fillna(median_col,inplace=True)

                    return df.isnull().sum()


                for column in x_columns:
                    replace_NA(x,column)
                    
                # Replace all x 'object' types Nan with median
                # imp = SimpleImputer(missing_values='NAN', strategy='most_frequent', fill_value=None)
                # new_col=imp.fit_transform(np.array(x['ocean_proximity']).reshape(-1, 1))

                # x['ocean_proximity']=new_col
                most_freq_cat=x['ocean_proximity'].value_counts()[0]#.to_frame()
                self.median_train_dict['ocean_proximity']=most_freq_cat
          
                x['ocean_proximity'].fillna(most_freq_cat,inplace=True)

                
                # Replace all y Nan with median
                if y is None:
                    pass
                else:
                    y_columns=y.columns
                    median_col_y=y["median_house_value"].median()
                    self.median_train_dict["median_house_value"]=median_col_y
                    y["median_house_value"].fillna(median_col_y,inplace=True)
                    y=y.to_numpy()

                #Storing one_hot encoded data to be used in validation set
                cat_columns = ["ocean_proximity"]
            
                #create the dummy variables
                df_processed = pd.get_dummies(x,prefix_sep="__",columns=cat_columns)

                #store the dummy variables from training dataset
                self.cat_dummies = [col for col in df_processed if "__" in col and col.split("__")[0] in cat_columns]
                #store all columns of the training dataset in a list
                self.processed_columns = list(df_processed.columns[:])

                #Normalize all the clean data
                self.min_train=df_processed.min()
                self.max_train=df_processed.max()
                x=(df_processed-self.min_train)/(self.max_train-self.min_train)


                
            else: #if data is test data
                
                x_columns=x.select_dtypes(include=['float']).columns
                # Replace all x Nan with median
                def replace_NA(df,column_name):
                    df[column_name].fillna(self.median_train_dict[column_name],inplace=True)

                    return df.isnull().sum()


                for column in x_columns:
                    null_values = replace_NA(x,column)
                
                
                # Replace all x 'object' types Nan with median
                x["ocean_proximity"].fillna(self.median_train_dict['ocean_proximity'],inplace=True)
                if y is None:
                    pass
                else:
                    y_columns=y.columns
                    y["median_house_value"].fillna(self.median_train_dict["median_house_value"],inplace=True)
                    y=y.to_numpy()

                #Convert all the clean data to numerical data
                # one_hot_encoded_data=pd.get_dummies(x, columns = ['ocean_proximity'])
                
                #one hot encode the test data set
                cat_columns = ["ocean_proximity"]

                df_test_processed = pd.get_dummies(x, prefix_sep="__", columns=cat_columns)

                #removing additional features from test set
                for col in df_test_processed.columns:
                    if ("__" in col) and (col.split("__")[0] in cat_columns) and col not in self.cat_dummies:
                        df_test_processed.drop(col, axis=1, inplace=True)  

                #adding 0's to missing features in test set
                for col in self.cat_dummies:
                    if col not in df_test_processed.columns:
                        print(f"col {col} not in df_test_processed.columns")
                        df_test_processed[col] = 0

                #add all missing One hot encoders
                df_test_processed = df_test_processed[self.processed_columns]

                #Normalize all the clean data
                for column in self.processed_columns:
                    df_test_processed[column] -= self.min_train[column]
                    df_test_processed[column] /= (self.max_train[column] - self.min_train[column])
                
                x = df_test_processed

            x=x.to_numpy()
            return x, (y if isinstance(y, (np.ndarray, np.generic,pd.DataFrame)) else None)


            #######################################################################
            #                       ** END OF YOUR CODE **
            #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        header_template = "Training a Regressor model with: x shape: {}, y shape: {}, neurons: {}, activations: {}, lr: {}, batch-size: {}, nb_epochs: {}"
        print(header_template.format(x.shape, y.shape, self.neurons, self.activations, self.learning_rate, self.batch_size, self.nb_epoch))
   

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        # Transform numpy arrays into tensors
        X = Variable(torch.from_numpy(X).type(dtype=torch.FloatTensor)).requires_grad_(True)
        Y = Variable(torch.from_numpy(Y).type(dtype=torch.FloatTensor)).requires_grad_(True)

        for epoch in range(self.nb_epoch):
            train_loss = 0
            avg_y_pred = 0
            samples = 0
            self.model.train()

            for (batch_X, batch_Y) in self._next_batch(X, Y, self.batch_size):
                # Zero the gradients
                self.optimiser.zero_grad()
                
                # Compute a forward pass
                output = self.model(batch_X) 

                # add the sum of y_pred to compute the avg
                avg_y_pred += output.sum().item()
                
                # Compute MSE based on forward pass
                loss_forward = self.mse_loss(output, batch_Y)
            
                # Perform backward pass 
                loss_forward.backward()

                # Update parameters of the model
                self.optimiser.step()
                train_loss += loss_forward.item() * batch_Y.size(0)
                samples += batch_Y.size(0)

            if (epoch)%10 == 0:
                train_template = "epoch: {} train rmse: {:e} avg y pred: {:e}"
                print(train_template.format(epoch, np.sqrt(train_loss/samples),
                    avg_y_pred / samples))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False)
        print(f"Predicting Y for X of shape {X.shape}")

        X = Variable(torch.from_numpy(X).type(dtype=torch.float32 ))
        output = self.model(X)
        return output.detach().numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y, training = False) # Do not forget
        print(f"Scoring with x of shape {X.shape} and y of shape {Y.shape}")

        y_model = self.predict(x)        
        
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


        return mean_squared_error(Y, y_model, squared=False) # Replace this code with your own

    def _next_batch(self, inputs, targets, batchSize):
        """ Generate the batches for training iteratively."""
        # loop over the dataset
        for i in range(0, inputs.shape[0], batchSize):
            # yield a tuple of the current batched data and labels
            yield (inputs[i:i + batchSize], targets[i:i + batchSize])

    def _weights_init(self, m):
        """ Initialize the weights with xavier uniform distribution.
        
        Args:
            m (torch.nn): layer of a neural network
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

def save_regressor(trained_model, model_path): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    try : 
        with open(model_path, 'wb') as target:
            pickle.dump(trained_model, target)
        print(f"\nSaved model in {model_path}\n")
    except Exception:
        traceback.print_exc()


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    try : 
        with open('part2_model.pickle', 'rb') as target:
            trained_model = pickle.load(target)
        print("\nLoaded model in part2_model.pickle\n")
        return trained_model
    
    except Exception:
        traceback.print_exc()



def RegressorHyperParameterSearch(x, y, hyperparameters): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.
    
    This function 

    Arguments:
        x (pd.DataFrame): raw input dataset
        y (pd.DataFrame): raw output data
        hyperparameters (dict[list]) dictionary containing the search range
            for each hyperparamter: neurons, activations, learning_rate,
            nb_epochs and batcch_size.
        
    Returns:
        results (list[dict]): evaluation metrics obtained for each combination
            of hyperparameters. Each element of the list is a dictionary containing
            the set of hyperparameters and the RMSE and the R2-score for the
            training and validation set.
            
            This is used to create the plots in our report. The optimal hyperparameters
            correspond to the element in the list with the lowest validation RMSE score.
            
        best_model (Regressor): model trained with the hyperparameters that
            minimize validation error.
        

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    
    # Split into train-val to perform hyperparameter tuning
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.1, random_state=42)

     # get list of all possible combintations of hyperparameters
    hyperparam_list = []
    num_experiments = len(hyperparameters['neurons'])
    for i in range(num_experiments):
        hyperparam_list.append({
            'neurons': hyperparameters['neurons'][i],
            'activations':hyperparameters['activations'][i],
            'batch_size':hyperparameters['batch_size'][i],
            'nb_epoch': hyperparameters['nb_epoch'][i],
            'learning_rate': hyperparameters['learning_rate'][i],
        })
    results = []
    best_error = float('inf')
    
    for param in hyperparam_list:
        result_dict = {}
        
        # define a regressor with the given hyperparameters
        regressor = Regressor(
            x_train, neurons = param['neurons'], activations = param['activations'], 
            batch_size = param['batch_size'], nb_epoch = param['nb_epoch'], 
            learning_rate = param['learning_rate'])
        
        # train the regressor with the validation data
        regressor.fit(x_train, y_train)
        
        # get train and test score
        train_rmse = regressor.score(x_train, y_train)
        val_rmse = regressor.score(x_val, y_val)
        
        # check if it is better than the best model
        if val_rmse < best_error:
            best_model = regressor
            best_error = val_rmse
            
        result_dict['rmse_train'] = train_rmse
        result_dict['rmse_val'] = val_rmse
        result_dict['params'] = param
        
        # get r2 scores
        y_train_pred = regressor.predict(x_train)
        y_val_pred = regressor.predict(x_val)
        result_dict['r2_train'] = r2_score(y_train, y_train_pred)
        result_dict['r2_val'] = r2_score(y_val, y_val_pred)
        
        print(f"Results: {result_dict}")
        
        results.append(result_dict)
    
    return results, best_model# Return the chosen hyper parameters


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

def depth_tuning():
    """ Tune the network depth.
    
    This function searches for the optimal network depth in the range [1,10]
    using the RegressorHyperParameterSearch function.
    
    The results are saved in `results_depth.pkl`, and the model
    trained with the optimal hyperparameters is saved in the path:
    `part2_model_tuned_depth.pickle`.
    """
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    
    
    depths = list(range(10))
    # define grid to search 
    hyperparameters = {
        'neurons': [[32]*d+[1] for d in depths],
        'activations': [["relu"]*(d+1) for d in depths],
        'batch_size': [128]*len(depths),
        'nb_epoch': [2000]*len(depths),
        'learning_rate': [0.01]*len(depths)
    }
    
    results, best_model = RegressorHyperParameterSearch(x, y, hyperparameters)
    
    # Save results and best model
    pickle.dump(results, open("results_depth.pkl", "wb"))
    save_regressor(best_model, "part2_model_tuned_depth.pickle")
    
def neurons_tuning():
    """ Tune number of neurons in the hidden layers.
    
    This function uses the optimal depth  found with the  previous functions,
     and finds the best number of neurons in the range ([12,136]) with 
     the RegressorHyperParameterSearch function.
    
    The results are saved in `results_neurons.pkl`, and the model
    trained with the optimal hyperparameters is saved in the path:
    `part2_model_tuned_neurons.pickle`.
    """
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    
    
    neuron_tests=[12,24,32,48,64,72,96,112,128,136]

    #change this to 2
    optimal_dept=3

    # define grid to search 
    hyperparameters = {
        'neurons': [[n]*optimal_dept+[1] for n in neuron_tests],
        'activations': [["relu"]*(optimal_dept+1) for n in neuron_tests],
        'batch_size': [128]*len(neuron_tests),
        'nb_epoch': [2000]*len(neuron_tests),
        'learning_rate': [0.01]*len(neuron_tests)
    }
    
    results, best_model = RegressorHyperParameterSearch(x, y, hyperparameters)
    
    # Save results and best model
    pickle.dump(results, open("results_neurons.pkl", "wb"))
    save_regressor(best_model, "part2_model_tuned_neurons.pickle") 


def lr_tuning():
    """ Tune learning rate.
    
    This function uses the optimal depth and number of neurons found with the
     previous functions, and finds the best learning rate
    in the range ([1e-8,1]) with the RegressorHyperParameterSearch function.
    
    The results are saved in `results_learning_rate.pkl`, and the model
    trained with the optimal hyperparameters is saved in the path:
    `part2_model_tuned_learning_rate.pickle`.
    """
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    
    
    neuron_optimal=48
    optimal_dept=3

    lr_tests = [1, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    # define grid to search 
    hyperparameters = {
        'neurons': [[neuron_optimal]*optimal_dept+[1]]*len(lr_tests),
        'activations': [["relu"]*(optimal_dept+1)]*len(lr_tests),
        'batch_size': [128]*len(lr_tests),
        'nb_epoch': [2000]*len(lr_tests),
        'learning_rate': lr_tests
    }
    
    results, best_model = RegressorHyperParameterSearch(x, y, hyperparameters)
    
    # Save results and best model
    pickle.dump(results, open("results_learning_rate.pkl", "wb"))
    save_regressor(best_model, "part2_model_tuned_learning_rate.pickle")   


def batch_size_tuning():
    """ Tune batch size.
    
    This function uses the optimal depth, number of neurons and learning
    rate found with the previous functions, and finds the best batch size
    in the range ([32,256]) with the RegressorHyperParameterSearch function.
    
    The results are saved in `results_epochs_batch_size.pkl`, and the model
    trained with the optimal hyperparameters is saved in the path:
    `part2_model_tuned_batch_size.pickle`.
    
    This function is the last step of our top-down hyperparameter search,
    and therefore the best model is also used as our final model, which
    is saved as: `part2_model.pickle`.
    """
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    
    
    neuron_optimal = 48
    optimal_dept = 3
    lr_optimal = 1e-3
    epochs_optimal = 2000 
    batch_size_test = [32, 64, 96, 128, 160, 192, 224, 256]

    # define grid to search 
    hyperparameters = {
        'neurons': [[neuron_optimal]*optimal_dept+[1]]*len(batch_size_test),
        'activations': [["relu"]*(optimal_dept+1)]*len(batch_size_test),
        'batch_size': batch_size_test,
        'nb_epoch': [epochs_optimal]*len(batch_size_test),
        'learning_rate': [lr_optimal]*len(batch_size_test)
    }
    
    results, best_model = RegressorHyperParameterSearch(x, y, hyperparameters)
    
    # Save results and best model
    pickle.dump(results, open("results_epochs_batch_size.pkl", "wb"))
    save_regressor(best_model, "part2_model_tuned_batch_size.pickle")   


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(
        x_train, neurons = [32, 32, 1], activations = ['relu', 'relu', 'relu'], 
        learning_rate = 0.01, nb_epoch = 2000, batch_size=128)


    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    print(x_train.shape)
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))



if __name__ == "__main__":
    example_main()
    # depth_tuning()
    # neurons_tuning()
    # lr_tuning()
    # batch_size_tuning()
    
    
