# Amazon Book Reviews: TF-IDF vs CountVectorizer with 8 Machine Learning Models

## Overview

This project builds an NLP text classification workflow on Amazon's Top 100 bestselling books and their customer reviews. It compares two text vectorization techniques, **TF-IDF** and **CountVectorizer**, across **8 machine learning classifiers**, with class imbalance handled through SMOTE and undersampling.

The notebook follows an end-to-end workflow:

1. Import libraries and load the two CSV files
2. Inspect structure, data types, and summary statistics
3. Perform exploratory data analysis (EDA)
4. Check missing values and impute them with KNN
5. Merge the books file with the reviews file
6. Clean and preprocess the review text
7. Apply stemming
8. Vectorize text with TF-IDF and with CountVectorizer
9. Handle class imbalance (SMOTE + RandomUnderSampler) inside pipelines
10. Train and evaluate 8 classifiers for each vectorizer
11. Compare accuracy across vectorizers and models

The project demonstrates the full path from raw book and review data → EDA → text cleaning → vectorization → imbalance handling → model comparison.

## Objective

Compare TF-IDF and CountVectorizer for text classification on Amazon book reviews, and find which classifier performs best with each.

## Problem Statement

Customer reviews are a rich source of text signal, but raw text has to be cleaned and converted into numbers before models can use it. Different vectorizers and algorithms can behave very differently on the same text.

The goal of this project is to build the same classification pipeline with two vectorizers and eight algorithms, then compare accuracy.

The text feature is built by joining `book title`, `review title`, and `review description`. The target variable is:

| Target | Description |
|---|---|
| Rank | Rank of the book in the Top 100 list (multi-class target) |

## Dataset

The dataset is **Top 100 Trending Books with Reviews** from Kaggle (scraped in November 2023). It has two CSV files.

### File 1: Top-100 Trending Books (100 rows, 8 columns)

| Column | Description |
|---|---|
| Rank | Rank in the Top 100 list |
| book title | Title of the book |
| book price | Book price |
| rating | Average rating (97 non-null values) |
| author | Book author |
| year of publication | Publication year (1947 to 2024) |
| genre | Book genre |
| url | Amazon link |

### File 2: Customer Reviews (920 rows, 10 columns)

| Column | Description |
|---|---|
| Sno | Row number |
| book name | Title of the reviewed book |
| review title | Title of the review |
| reviewer | Reviewer name |
| reviewer rating | Rating given by the reviewer (2 to 5) |
| review description | Full review text |
| is_verified | Whether the purchase is verified |
| date | Review date |
| timestamp | Review location and date text |
| ASIN | Amazon product ID |

## Workflow

### 1. Library Imports

**Data Processing**
- pandas
- numpy
- BeautifulSoup

**Visualization**
- matplotlib
- seaborn
- plotly (express and graph_objects)
- WordCloud

**Preprocessing**
- KNNImputer
- StandardScaler
- MinMaxScaler
- NLTK (stopwords, PorterStemmer)

**Text Vectorization**
- TfidfVectorizer
- CountVectorizer

**Imbalance Handling**
- SMOTE
- RandomUnderSampler
- imblearn Pipeline

**Machine Learning**
- scikit-learn classifiers and metrics

### 2. Exploratory Data Analysis

- Book title: value counts, highest ranked (*Iron Flame*), lowest ranked (*The Wonky Donkey*), plus a word cloud of titles
- Rating: mean 4.69, range 4.1 to 5.0
- Year of publication: 1947 to 2024, spanning 78 years
- Genre: 79 distinct genres, Nonfiction is the most common (6 books)
- Book name: reviews per book (some books have 10 or 20 reviews)
- Reviewer rating: mean 4.83, range 2 to 5
- is_verified: 917 verified and 3 non-verified reviews
- Timestamp: review dates from April 2023 onward

### 3. Missing Values

Only the `rating` column of the books file has missing values (3%). These are imputed with `KNNImputer`.

### 4. Data Merging

The books file and the reviews file are merged on book title with an inner join, giving **890 rows** for modeling.

### 5. Text Preprocessing

The `review description` column is cleaned step by step:

| Step | Purpose |
|---|---|
| Lower casing | Treat "Book" and "book" as the same word |
| Remove whitespace | Reduce noise |
| Remove HTML tags | Strip markup |
| Remove URLs | Remove irrelevant links |
| Remove punctuation | Normalize text |
| Remove special characters | Keep letters, digits, spaces |
| Remove chat words | Handle informal abbreviations |
| Remove emojis | Reduce noise |
| Remove stopwords | Keep meaningful words (NLTK English list) |
| Stemming | Reduce words to their base form (PorterStemmer) |

### 6. Feature and Target

- **X**: `book title, review title, review description`
- **y**: `Rank`

### 7. Vectorization

* Vectorizer

    * TfidfVectorizer 
    * CountVectorizer 

### 8. Class Imbalance Handling

The ranks are imbalanced across classes, so each pipeline includes:

- **SMOTE** to oversample minority classes 
- **RandomUnderSampler** to balance the result

### 9. Models

Eight classifiers are trained, each in a pipeline with the sampling steps:

1. Random Forest
2. Logistic Regression
3. Multinomial Naive Bayes
4. Support Vector Classifier (SVC)
5. Passive Aggressive Classifier
6. Gradient Boosting
7. K-Nearest Neighbors
8. Multilayer Perceptron (MLP)

Each model is evaluated with accuracy and a confusion matrix.

## Results

### TF-IDF

| Model | Accuracy |
|---|---|
| **K-Nearest Neighbors** | **0.983** |
| Multilayer Perceptron | 0.961 |
| Random Forest | 0.955 |
| Passive Aggressive | 0.955 |
| Gradient Boosting | 0.955 |
| Logistic Regression | 0.938 |
| Support Vector Classifier | 0.910 |
| Multinomial Naive Bayes | 0.893 |

### CountVectorizer

| Model | Accuracy |
|---|---|
| **Random Forest** | **0.955** |
| **Gradient Boosting** | **0.955** |
| Passive Aggressive | 0.949 |
| Multilayer Perceptron | 0.933 |
| Logistic Regression | 0.926 |
| Multinomial Naive Bayes | 0.865 |
| K-Nearest Neighbors | 0.735 |
| Support Vector Machine | 0.410 |

### Key Findings

- With **TF-IDF**, K-Nearest Neighbors performed best (0.983).
- With **CountVectorizer**, Random Forest and Gradient Boosting performed best (0.955).
- TF-IDF gave equal or better accuracy than CountVectorizer for every model, with the biggest gaps for KNN (0.983 vs 0.735) and SVM (0.910 vs 0.410).

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly, WordCloud |
| NLP | NLTK, BeautifulSoup, regex |
| Vectorization | TfidfVectorizer, CountVectorizer |
| Machine Learning | Scikit-learn |
| Imbalance Handling | imbalanced-learn (SMOTE, RandomUnderSampler) |

## Summary

This project implements a complete NLP classification workflow on Amazon's Top 100 bestselling books and their reviews. It cleans and stems the review text, vectorizes it with TF-IDF and CountVectorizer, balances classes with SMOTE and undersampling, and compares eight classifiers under each vectorizer.

The best result was **0.983 accuracy** (TF-IDF, K-Nearest Neighbors); the best CountVectorizer result was **0.955** (Random Forest and Gradient Boosting).
