# Twitter Sentiment Analysis Project

## Overview
This project focuses on analyzing sentiments expressed in tweets related to various topics, specifically political discourse. By utilizing Natural Language Processing (NLP) techniques, the project aims to classify tweets as positive, negative, or neutral based on their content. The analysis helps in understanding public sentiment and trends over time.

## Purpose
The primary purpose of this project is to:

- Analyze the sentiment of tweets to gauge public opinion.
- Visualize sentiment trends over time.
- Identify key themes and topics within the tweets.
- Provide insights into how sentiment changes in response to specific events.

## Goals
1. **Data Collection**: Read and preprocess a dataset of tweets for sentiment analysis.
2. **Data Cleaning**: Clean the text data by removing noise such as symbols, punctuation, and stop words.
3. **Sentiment Classification**: Map numerical sentiment values to categorical labels (positive, negative, neutral).
4. **Exploratory Data Analysis**: Analyze and visualize the sentiments to identify patterns and trends.
5. **Model Development**: Optionally develop a machine learning model to predict sentiment based on tweet content.

## Installation
To run this project, make sure you have Python installed along with the necessary libraries. You can install the required libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib nltk
```

## Usage
### Reading and Cleaning Data
The following code snippet reads the Twitter data from a CSV file and displays the first few rows:

```python
import pandas as pd

# Read data for Twitter sentiment analysis
df = pd.read_csv("Twitter_Data.csv")
print(df.head())
```

### Data Preparation
Prepare the data by mapping numerical categories to descriptive labels:

```python
# Change the dependent variable to categorical labels
df['category'] = df['category'].map({0.0: 'neutral', -1.0: 'negative', 1.0: 'positive'})
print(df.head())
```

### Missing Value Analysis
Analyze missing values and drop any rows with null values:

```python
# Check for missing values
print(df.isnull().sum())
df.dropna(inplace=True)
```

### Text Cleaning
Clean the text data by converting to lowercase, removing non-alphanumeric characters, and eliminating stop words:

```python
import nltk
from nltk.corpus import stopwords

# Convert text to lowercase
df['clean_text'] = df['clean_text'].str.lower()

# Remove non-alphanumeric characters
df['clean_text'] = df['clean_text'].str.replace(r'[^a-zA-Z\s]+', '')

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
```

### Analyzing Sentence Length
Create a new column to find the length of each cleaned sentence:

```python
df['length'] = df['clean_text'].apply(lambda row: len(row.split()) if isinstance(row, str) else None)
print(df.head())
```

### Visualization
Visualize the sentiment distribution using Seaborn:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize sentiment distribution
sns.countplot(x='category', data=df)
plt.title('Sentiment Distribution')
plt.show()
```

### Optional: Model Development
You can extend the project by developing a machine learning model to predict sentiment. This can be done using libraries like `scikit-learn` or `TensorFlow`.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Splitting the dataset
X = df['clean_text']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Training a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
```

## Conclusion
This project provides a comprehensive framework for analyzing sentiments in tweets. By cleaning the data, classifying sentiments, and visualizing the results, users can gain valuable insights into public opinion and trends. Further enhancements can include model development for automated sentiment prediction and more advanced NLP techniques.
