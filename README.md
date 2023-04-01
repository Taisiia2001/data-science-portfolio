# data-science-portfolio
This repository contains the code and resources for my personal data science projects. 

---

# Star Type Classification using Logistic Regression

In this project, I developed a logistic regression model to classify different types of stars based on their physical attributes. The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/deepu1109/star-dataset).

## Dataset

The dataset contains several features of 240 stars, including Absolute Temperature (in K), Relative Luminosity (L/Lo), Relative Radius (R/Ro), Absolute Magnitude (Mv), Star Color (white, Red, Blue, Yellow, yellow-orange, etc), Spectral Class (O, B, A, F, G, K, M), and Star Type (Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence , SuperGiants, HyperGiants). The goal of the project is to build a model that is trained on first six features (X) to classify the Star Type variable (y)

## Approach

First, I performed some exploratory data analysis (EDA) and did minor data cleaning on Star Color feature. Then, I preprocessed the data by encoding categorical variables into their numeric representation and splitting the dataset into training and testing sets.

Then, I trained a logistic regression model on the training data and evaluated its performance on the test data. I also used SHAP (SHapley Additive exPlanations) to interpret the model's predictions and gain insights into which features are most important for the classification task.

## Results

My logistic regression model achieves an accuracy of approximately 95% on the test set, indicating that it is able to accurately classify different types of stars based on their physical attributes. I found that the most important features for this particular classification task are the surface temperature and the absolute magnitude of the stars.

## Tools & Libraries

The following Python libraries were used in this project:

- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- SHAP

---

# LDA Topic Analysis of Political Subreddits

In this project, I used Latent Dirichlet Allocation (LDA) to analyze the topics of Reddit posts on Liberal and Conservative subreddits. LDA is a statistical model that allows for the identification of latent topics in a corpus of text data. The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/neelgajare/liberals-vs-conservatives-on-reddit-13000-posts)

## Dataset

The dataset used in this project contains 13000 rows of Reddit posts collected from Liberal and Conservative leaning subreddits.

## Approach

My first step was to preprocess the data, which involved removing stop words, lemmatizing the text, and creating a bag of words representation of the data. Then I trained an LDA model on the preprocessed data using Gensim library. The optimal number of topics was determined using a coherence score. Finally, I used t-SNE to cluster my topics. I used matplotlib and bokeh to visualize clusters. 

## Results

The LDA model identifies several human-interpretable topics, including war in Ukraine, American elections, and protests in Canada. The t-SNE clusters reveal patterns in the data that correspond to the different topics.  

## Tools & Libraries

The following Python libraries were used in this project:

- Pandas
- Gensim
- spaCy
- Scikit-learn
- Matplotlib
- Bokeh

---

# EEG data classification with CNN

This project aims to classify EEG data into different mental sates using a convolutional neural network (CNN) architecture. EEG (electroencephalogram) is a non-invasive technique to record brain activity, which can be used to detect different mental states such as relaxed, focused, or meditative.

## Dataset

The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition) and consists of 179 columns: explanatory variables X1, X2... X178, where each data point is the value of the EEG recording at a different point in time, and response variable y in column 179 that contains the category of the 178-dimensional input vector. The five categories of the EEG recordins represent the following states: 

- 5 - eyes open, means when they were recording the EEG signal of the brain the patient had their eyes ope
- 4 - eyes closed, means when they were recording the EEG signal the patient had their eyes closed
- 3 - they identify where the region of the tumor was in the brain and record the EEG activity from the healthy brain area
- 2 - they record the EEG from the area where the tumor was located
- 1 - recording of seizure activity

## Approach

First, I split my data into train and test sets and reshaped it into 3D array to fit the input shape of CNN. Then, I converted labels to one-hot encoded vectors and fit the model.

The CNN model used in this project has the following architecture:

- Convolutional layer with 32 filters of size 3x3 and ReLU activation
- Max pooling layer with pool size 2x2
- Convolutional layer with 64 filters of size 3x3 and ReLU activation
- Max pooling layer with pool size 2x2
- Flatten layer
- Dense layer with 128 neurons and ReLU activation
- Droput layer with dropout raate of 0.5
- Output layer with 5 neurons and softmax activation

The model was trained using a categorical crossentropy loss function and Adam optimizer. 

## Results

The CNN model achieved an accuracy of 72% on the test set, which is a moderate result for this classification task. Future work on this project could include exploring different CNN architecture to improve the accuracy score and investigating the interpretability of the model using techniques such as layer visualization and saliency maps. 

## Tools & Libraries

- Pandas
- NumPy
- Scikit-learn
- Keras
- Plotly
