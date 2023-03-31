import numpy as np
import pickle
import traceback
from numpy.random import default_rng


def xavier_init(size, gain = 1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative 
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """ 
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        """
        Numpy sigmoid activation implementation
        Arguments:
        x - numpy array of any shape
        Returns:
        A - output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
        """

        self._cache_current=x
        return 1/(1+np.exp(-x))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################


        """
        The backward propagation for a single SIGMOID unit.
        Arguments:
        grad_z - post-activation gradient, of any shape
        Returns:
        dZ - Gradient of the cost with respect to Z
        """



        sig=self.forward(self._cache_current)

        #grad_z=dL/dh
        return grad_z*sig*(1-sig)


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._cache_current=x
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        return np.maximum(0,x)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################



        """
        The backward propagation for a single RELU unit.
        Arguments:
        grad_z - post-activation gradient, of any shape
        Returns:
        dZ - Gradient of the cost with respect to Z
        """
        dZ = np.array(grad_z, copy = True)
        dZ[self._cache_current <= 0] = 0

        # dZ[self._cache_current > 0] = 1
   
        return dZ

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        self._W = xavier_init(size=(n_in, n_out))
        self._b = np.random.rand(1, n_out) - 0.5

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None


    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        self._cache_current = (x, self._W, self._b)
        return np.dot(x, self._W) + self._b

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        x_previous = self._cache_current[0]
        W_previous = self._cache_current[1]
        self._grad_W_current = np.dot(x_previous.T, grad_z)

        self._grad_b_current = np.sum(grad_z, axis=0)


        return np.dot(grad_z, W_previous.T)
        

       
    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        self._W -= learning_rate * self._grad_W_current
        self._b -= learning_rate * self._grad_b_current
        


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding 
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._layers = []
        for layer_num in range(len(self.neurons)):
            # add linear layer
            n_out = self.neurons[layer_num]
            if layer_num == 0:
                n_in = self.input_dim
            else:
                n_in = self.neurons[layer_num-1]
            self._layers.append(LinearLayer(n_in, n_out))
            
            # add activation function
            if self.activations[layer_num] == "relu":
                self._layers.append(ReluLayer())
            else:
                self._layers.append(SigmoidLayer())
            
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if len(np.shape(x)) == 1:
            x = np.reshape(x,(np.shape(x)[0],1))
        y = x
        # propagate x forward through the network
        for layer in self._layers:
            y = layer.forward(y)
        return y 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # propagate grad_z backwards through the network
        grad = grad_z
        for layer in self._layers[::-1]:
            # get grad of loss wrt layer input
            grad = layer.backward(grad)

        
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in self._layers:
            layer.update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                cross_entropy.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._loss_layer = None
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns: 
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        assert input_dataset.shape[0] == target_dataset.shape[0], f'The number of x instances should match y instances'
        if len(np.shape(input_dataset)) == 1:
            input_dataset = np.reshape(input_dataset,(np.shape(input_dataset)[0],1))

        seed = 60012
        rg = default_rng(seed)
        shuffled_indices = rg.permutation(input_dataset.shape[0])
        return input_dataset[shuffled_indices], target_dataset[shuffled_indices]
        # except Exception:
            # traceback.print_exc()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        

        try:
            if len(np.shape(input_dataset)) == 1:
                input_dataset = np.reshape(input_dataset,(np.shape(input_dataset)[0],1))
            if len(np.shape(target_dataset)) == 1:
                target_dataset = np.reshape(target_dataset,(np.shape(target_dataset)[0],1))
                    
            dataset = np.append(input_dataset, target_dataset,axis=1)  
            input_shape = np.shape(input_dataset)[1]

            for epoch in range(self.nb_epoch):
                if self.shuffle_flag:
                    shuffled_inputs, shuffled_targets = self.shuffle(input_dataset, target_dataset)
                    dataset= np.append(shuffled_inputs, shuffled_targets,axis=1)
                batches = np.array_split(dataset, self.batch_size)
                
                for batch in batches :
                    output = self.network(batch[:,:input_shape])
                    if self.loss_fun == "mse":
                        loss_f = CrossEntropyLossLayer()
                    else:
                        loss_f = MSELossLayer()

                    loss = loss_f.forward(output, batch[:,input_shape:])
                    grad_z = loss_f.backward()
                    self.network.backward(grad_z)
                    self.network.update_params(self.learning_rate)

        except Exception:
            traceback.print_exc()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data. Returns
        scalar value.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).
        
        Returns:
            a scalar value -- the loss
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        try:
            if len(np.shape(input_dataset)) == 1:
                input_dataset = np.reshape(input_dataset,(np.shape(input_dataset)[0],1))

            if self.loss_fun == "mse":
                loss_f = CrossEntropyLossLayer()
            else:
                loss_f = MSELossLayer()

            loss = loss_f.forward(self.network(input_dataset), target_dataset)

            return loss

        except Exception:
            traceback.print_exc()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        #clean the data
        col_median =np.nanmedian(data,axis=0)


        inds = np.where(np.isnan(data))
        # #Place column means in the indices. Align the arrays using take
        data[inds] = np.take(col_median, inds[1])

        self.col_max = data.max(axis=0)
        self.col_min = data.min(axis=0)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # col_max = data.max(axis=0)
        # col_min = data.min(axis=0)
        np.seterr(divide='ignore', invalid='ignore')

        normalized_data=(data-self.col_min)/(self.col_max-self.col_min)
        normalized_data[np.isnan(normalized_data)] = 0

        return normalized_data

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retrieve the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        np.seterr(divide='ignore', invalid='ignore')

        reverted_data=(data*(self.col_max-self.col_min))+self.col_min
        reverted_data[np.isnan(reverted_data)] = 0
        return reverted_data

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)
    x = dat[:, :4] 

    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)


    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)

    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))



    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()

    #Linear layer
    # # Start
    # inputs = np.ones((8, 3))
    # grad_loss_wrt_outputs = np.ones((8, 42))
    # layer = LinearLayer(n_in=3, n_out=42)
    # outputs = layer(inputs)
    # grad_loss_wrt_inputs = layer.backward(grad_z=grad_loss_wrt_outputs)
    # # End


    #Testing the Preprocessor
    # #Start
    # print(f' ----Testing the Preprocessor----')


    # dat = np.loadtxt("iris.dat")
    # np.random.shuffle(dat)

    # x = dat[:, :4]
    # y = dat[:, 4:]

    # split_idx = int(0.8 * len(x))#5

    # x_train = x[:split_idx]
    # y_train = y[:split_idx]
    # x_val = x[split_idx:]
    # y_val = y[split_idx:]

    # prep_input = Preprocessor(x_train)

    # x_train_pre = prep_input.apply(x_train)


    # print(f' normalized training data \n {x_train_pre}')
    # x_train_pre_revert = prep_input.revert(x_train_pre)
    # print(f' revert normalized training data \n {x_train_pre_revert}')
    # test_normalization=prep_input.revert(prep_input.apply(x_train))
    # print(f' normalized training data (embed) \n {test_normalization}')
    #End test preprocessor
