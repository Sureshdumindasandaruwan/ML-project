# ML-project
G02(2020s18393)

## Description

### Preprocessing
Here's a breakdown of the key steps in the code:

- **Data Importing**: The dataset `used_device_data.csv` is imported using pandas, and initial exploration is done to understand the data's structure and contents.
- **Data Preprocessing**: Symbols in the 'os' column are replaced with missing values, and missing values in the dataset are handled by replacing them with median or mode values, depending on the distribution of the data.
- **Data Visualization**: Various plots are created to visualize the data distribution, including histograms, box plots, distribution plots, heatmaps, count plots, and scatter plots.
- **Outlier Handling**: Outliers in the dataset are identified and handled by capping them to certain quantile values.
- **Feature Engineering**: Categorical columns are converted to numerical columns using one-hot encoding, and the data is normalized.

The code uses common data science libraries such as pandas, NumPy, matplotlib, seaborn, and scikit-learn to perform these tasks. It's a typical workflow for preparing a dataset for further analysis or machine learning model training.

### Neural Network Model Fitting

- **Model Definition**: A deep neural network (DNN) model is defined using a sequence of layers. Each layer specifies the number of neurons and the activation function to be used.

- **Compilation**: The model is compiled, which includes specifying the loss function, optimizer, and metrics for evaluation.

- **Training**: The model is trained on the preprocessed data by calling the `fit` method, where it learns to map inputs to outputs. This step involves feeding the training data into the model and adjusting the model parameters (weights and biases) through backpropagation.

- **Evaluation**: After training, the model's performance is evaluated on a separate test set to determine its accuracy and generalization capability.

- **Hyperparameter Tuning**: Various hyperparameters such as learning rate, number of epochs, and batch size are tuned to improve the model's performance.

- **Model Saving**: Once the model is trained and tuned, it is saved for future use or deployment.

This process is iterative and may involve multiple rounds of training and tuning to achieve the desired performance. The code would use TensorFlow, Keras, or a similar library to implement these steps.


### User Manual: Neural Network Hyperparameter Tuning GUI

**1. Introduction**

Welcome to the Neural Network Hyperparameter Tuning GUI! This graphical user interface (GUI) allows you to manually adjust hyperparameters and visualize the training and validation mean absolute error (MAE) of a neural network model.

**2. Getting Started**

To use the GUI, follow these steps:

- Make sure you have Python installed on your system.
- Install the required libraries by running ***pip install numpy matplotlib scikit-learn tensorflow.***

**3. GUI Overview**

The GUI consists of several elements:

- **Number of Hidden Layers:** Use the spinbox to specify the number of hidden layers in the neural network.
- **Optimizer:** Choose the optimizer for model training from the dropdown menu.
- **Epochs:** Set the number of training epochs using the spinbox.
- **Batch Size:** Select the batch size for training from the dropdown menu.
- **Neurons and Activation Function:** Enter the number of neurons and select the activation function for each hidden layer.
- **Buttons:** Use the *"Enter"* button to specify the parameters for each hidden layer and the *"Next Layer"* button to move to the next layer. The *"Train"* button trains the model, the *"Plot"* button visualizes the training and validation MAE, the *"Clear"* button clears the plot and resets the inputs, and the *"Save"* button saves the plot as an image (currently disabled).

**4. Usage**

Follow these steps to use the GUI:

1. Enter the number of hidden layers in the provided spinbox.
2. Select the optimizer, specify the number of epochs, and choose the batch size.
3. For each hidden layer, enter the number of neurons and select the activation function. Click "Next Layer" to move to the next layer.
4. After specifying all hidden layers, click "Train" to train the model.
5. Once training is complete, click "Plot" to visualize the training and validation MAE.
6. To clear the plot and reset inputs, click "Clear".
7. (Optional) Click "Save" to save the plot as an image (currently disabled).

**5. Notes**

Ensure that you have the necessary data prepared before training the model.
Experiment with different hyperparameters to find the best configuration for your neural network.

