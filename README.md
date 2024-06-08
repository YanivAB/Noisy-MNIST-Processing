# CNN Model for Noisy MNIST Dataset
This project aims to improve the classification accuracy of a Convolutional Neural Network (CNN) on the MNIST dataset with artificially added noise. The project includes preprocessing steps to reduce the noise and improve the model's performance.

# Introduction
The MNIST dataset is a well-known benchmark for evaluating image classification algorithms. In this project, the dataset has been modified to include noise, and various preprocessing techniques are applied to mitigate the noise and improve classification accuracy.

## Documentation
The project was maded for Technion Signal Processing course (035039=) by Dr. Igor Demchenko and Dr. Miri Benyamini.

## Credits
- Developed by Yaniv Abramov, Michael Ben Ezra and Idan Dror.

# Dataset
The dataset consists of two main files:
mnist_noisy_train.csv: Training data with noisy images.
mnist_noisy_test.csv: Test data with noisy images.
These files can be found here:
https://technionmail-my.sharepoint.com/:f:/g/personal/yanivabramov_campus_technion_ac_il/EreOFMk1o4xLoan_bNWsY5QBCeI6MCUfx_XmrVVecQ1xFA?e=WdWhaT 

# Preprocessing
Preprocessing steps include:

Spectral Density Analysis: Analyzing the spectral density of noise in the images.
Data Cleaning: Applying filters to reduce noise.
PCA: Using Principal Component Analysis (PCA) to reduce dimensionality while retaining 90% of the variance.
Averaging: Averaging images of the same class to reduce noise.
# Model
A Convolutional Neural Network (CNN) is used for classification. The model is provided in kaggle repository: https://www.kaggle.com/code/heeraldedhia/mnist-classifier-first-deep-learning-project

# Results
The results of the preprocessing and model performance are documented, showing improvements in classification accuracy after noise reduction.

# Requirements
To run this project, you need to:
1. Install dependencies: pip install -r requirements.txt
2. pip install numpy pandas matplotlib keras scipy scikit-image opencv-python requests

# Usage
Download the Dataset: Use the provided script to download the dataset from OneDrive.
Preprocess the Data: Run the preprocessing scripts to clean and prepare the data.
Train the Model: Use the provided CNN model script to train on the preprocessed data.
Evaluate the Model: Evaluate the model on the test data and analyze the results.

# License
This project is licensed under the MIT License - see the LICENSE file for details.



# Pratameter

# noisy parameters:
train_path, test_path: path to noisy mnist csv files. str   #you can change it to any noisy mnist data.
X_train_ns, y_train_ns, X_test_ns, y_test_ns: converted csv's data of the noisy mnist images. type-np.ndarray
X_train_ns shape (60000, 784), y_train_ns shape (60000,), X_test_ns shape (10000, 784), y_test_ns shape (10000,)

X_train_mat, X_test_mat: matrix form of images. type np.ndarray
X_train_mat shape(60000, 28,28) , X_test_mat shape(10000, 28,28)


# clean parameters:
x_mnist, y_mnist: MNIST train x and y from MNIST API. type- np.ndarry
x_mnist shape (60000, 28, 28), y_mnist shape (60000,)
x_mnist_test, y_mnist_test: MNIST test x and y from MNIST API. type- np.ndarry
x_mnist_test shape (10000, 28, 28), y_mnist_test shape (10000,)


# Denoised parameters (gaussian):

X_filtered_train   (60000, 28, 28)
X_filtered_test  (10000, 28, 28)