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
