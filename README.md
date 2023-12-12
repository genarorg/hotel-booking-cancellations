# Predicting Hotel Booking Cancellations.

## Introduction
EDA and Data preparation are handled in a local notebook file, but the model analysis is done in a Google Colab notebook:

[EDA and Data Preparation](src/hotel-booking-cancellations.ipynb)
[Model Building and Analysis](https://colab.research.google.com/drive/15ZaPohqmvR2tk3uvB2OW6NwEVyM9blpW?usp=sharing)

### 1. Research Question

Our primary research question is: "How can we predict the likelihood of a hotel booking being canceled based on available booking data?"

We will explore this question by analyzing a dataset of hotel bookings from two hotels in Portugal. The dataset includes information 
such as booking dates, length of stay, number of guests, and booking cancellations. We will use this data to develop a predictive model 
that can accurately identify the likelihood of a booking being canceled.

### 2. Data Sources

The dataset for this analysis is sourced from one hotel in Portugal and includes detailed booking information, including cancellation status. 

This data is available at [https://www.sciencedirect.com/science/article/pii/S2352340918315191#bib5], and a PDF copy of the article is included.

The dataset contains 2 files, but this work will focus only on one of the files, `H1.csv`.

### 3. Analysis Techniques

For this project, we plan to employ various machine learning techniques, including:

* Exploratory Data Analysis (EDA) to understand the dataset's characteristics.
* Feature Engineering to extract and select relevant features influencing cancellations.
* Predictive Modeling using Logistic Regression, Random Forest, and SVM to predict booking cancellations.
* Evaluation of model performance using metrics like accuracy, precision, recall, and AUC-ROC.

### 4. Expected Results

We anticipate developing a predictive model that can accurately identify the likelihood of a booking being canceled. 
The model's effectiveness will be measured by its `accuracy`, `precision` and `f1` scores.

### 5. Importance of the Question

Understanding and predicting hotel booking cancellations are vital for effective revenue management in the hospitality 
industry. By accurately forecasting cancellations, hotels can optimize occupancy rates, adjust pricing strategies, and 
improve overall customer service. This analysis aims to provide actionable insights that can lead to more efficient 
hotel management and enhanced customer satisfaction.

