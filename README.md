# Fruit Classification and Clustering
## Project Overview
This project involves the classification and clustering of fruit data using various machine learning algorithms. The dataset includes various features related to fruit shapes and sizes, which are used to train and evaluate multiple models, including a neural network, an MLP classifier, and several clustering algorithms.

### Dataset
The dataset is an Excel file (Date_Fruit_Datasets.xlsx) that contains 898 samples with 34 features per sample. The features include various geometrical and statistical properties of the fruits, and the target variable is the class label of the fruit.

### Data Preprocessing
Loading Data: The data is loaded from an Excel file using pandas.
Label Encoding: The target variable 'Class' is encoded using LabelEncoder to convert categorical class labels into numerical values.

### Models

#### 1. Neural Network (TensorFlow/Keras)
Architecture:

Input layer: 50 neurons, ReLU activation
Hidden layers: 30 neurons, ReLU activation; 15 neurons, ReLU activation (repeated twice)
Output layer: 7 neurons, Softmax activation

Compilation:
Optimizer: RMSprop
Loss function: Sparse categorical cross-entropy
Metrics: Accuracy

Training:
Epochs: 100
Validation data: Used to evaluate performance during training

Evaluation:
Loss and accuracy are plotted for both training and validation data.
Confusion matrix and classification report are generated for test data.

#### 2. Multi-layer Perceptron Classifier (scikit-learn)
Training and Evaluation:
The MLP classifier is trained and evaluated with the dataset.
Classification report and confusion matrix are generated to assess performance.

#### 3. Clustering Algorithms
Algorithms Used:
K-Means
Hierarchical Clustering
DBSCAN
Gaussian Mixture Models

### Evaluation:
Clustering results are evaluated based on various metrics and visualizations.

### Conclusion
This project demonstrates the application of machine learning techniques for fruit classification and clustering, providing insights into model performance and clustering effectiveness.
