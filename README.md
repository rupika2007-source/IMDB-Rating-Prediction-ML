# 🎬 IMDb Rating Prediction — Machine Learning Project

> **Can we teach a computer to predict how audiences will rate a movie on IMDb using data available before and after release?**

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Feature Engineering](#feature-engineering)
- [Models Trained](#models-trained)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Team](#team)

---

## Overview

This project performs end-to-end Exploratory Data Analysis (EDA) and machine learning on IMDb movie data to predict audience ratings. We train and compare **8 regression models** — from simple linear regression to tuned gradient boosting ensembles — and build two distinct prediction systems: one using **post-release data** and one using only **pre-release information**.

Our best model (Tuned Gradient Boosting) achieves an **R² of 0.82** with an average prediction error of just **0.20 rating points** on a 10-point scale.

---

## Problem Statement

> *Can we teach a computer to predict how audiences will rate a movie on IMDb — using data available before and after a film's release?*

We approach this in **two ways**:

| Scenario | Features Used | Goal |
|---|---|---|
| **Post-release** | Critic scores, votes, box office, runtime, director & genre reputation | High-accuracy rating prediction |
| **Pre-release** | Director, genre, runtime, certificate only | Early prediction before a film releases |

---

## Dataset

We merged two publicly available IMDb datasets:

| Dataset | Rows | Columns |
|---|---|---|
| IMDb Top 1,000 | 1,000 | 16 |
| IMDb Top 10,000 | 9,999 | 12 |
| **After merge & deduplication** | **9,861** | **6 final features** |

**Target variable:** `IMDB_Rating` (continuous float, scale 1.0–10.0)

**Raw features used:**

| Feature | Type | Description |
|---|---|---|
| `Meta_score` | Numeric | Metacritic critic score (0–100) |
| `No_of_Votes` | Numeric | Number of user votes on IMDb |
| `Runtime` | Numeric | Film duration in minutes |
| `Gross` | Numeric | Box office earnings (USD) |
| `Director` | Categorical | Director name |
| `Genre` | Categorical | Primary genre of the film |
| `Certificate` | Categorical | Age rating (PG, R, U/A, etc.) |

---

## Project Pipeline

```
Load Datasets  →  Standardise Columns  →  Merge & Deduplicate
      ↓
Clean Data (nulls, type conversion, string parsing)
      ↓
Exploratory Data Analysis (distributions, correlations, genre & director analysis)
      ↓
Feature Engineering (Director_Avg_Rating, Genre_Avg_Rating, Log transforms)
      ↓
Model Training (8 models: Linear → Ridge → Lasso → RF → GBR → Ensemble)
      ↓
Hyperparameter Tuning (GridSearchCV, 5-fold cross-validation)
      ↓
Evaluation (R², RMSE, MAE) + Feature Importance Analysis
      ↓
Pre-Release Model (Director, Genre, Runtime, Certificate only)
      ↓
predict_imdb_rating() — production-ready prediction function
```

---

## Feature Engineering

Four new features were engineered from the raw data, all of which outperformed the original columns in predictive power:

| Feature | How Created | Why |
|---|---|---|
| `Director_Avg_Rating` | Mean IMDb rating of all films by the same director | Captures director reputation and track record |
| `Genre_Avg_Rating` | Mean IMDb rating of all films in the same primary genre | Captures genre baseline expectations |
| `Log_Votes` | `log(1 + No_of_Votes)` | Compresses heavily skewed vote distribution |
| `Log_Gross` | `log(1 + Gross)` | Compresses heavily skewed box office distribution |

> **Why log-transform?** Both votes and gross are right-skewed — a few blockbusters dominate. Log-transform makes the distribution normal and the relationship with ratings linear, which significantly improves model accuracy.

---

## Models Trained

| Model | Type | Notes |
|---|---|---|
| Linear Regression | Baseline | No regularisation |
| Ridge Regression | L2 regularisation | Prevents overfitting from correlated features |
| Lasso Regression | L1 regularisation | Feature selection via coefficient shrinkage |
| Random Forest | Ensemble (parallel trees) | 100 estimators, handles non-linearity |
| Tuned Random Forest | GridSearchCV optimised | 5-fold CV across 24 hyperparameter combinations |
| Gradient Boosting | Ensemble (sequential trees) | Each tree corrects errors of the previous |
| Tuned Gradient Boosting | GridSearchCV optimised | **Best model** |
| Combined (RF + GBR) | Simple average ensemble | Averages predictions of both tuned models |

---

## Results

### Post-Release Model Performance

| Model | R² | RMSE | MAE | Rank |
|---|---|---|---|---|
| **Tuned Gradient Boosting** | **0.82** | **0.28** | **0.20** | 🥇 1st |
| Combined (RF + GBR) | 0.81 | 0.29 | 0.21 | 🥈 2nd |
| Gradient Boosting | 0.80 | 0.30 | 0.22 | 3rd |
| Tuned Random Forest | 0.79 | 0.31 | 0.23 | 4th |
| Random Forest | 0.77 | 0.32 | 0.24 | 5th |
| Ridge Regression | 0.62 | 0.41 | 0.31 | 6th |
| Linear Regression | 0.61 | 0.42 | 0.32 | 7th |
| Lasso Regression | 0.61 | 0.42 | 0.32 | 8th |

> **R²** = proportion of variance explained (1.0 = perfect).
> **RMSE** = average error in rating points (same unit as the target).
> **MAE** = mean absolute error — our best model is off by just **0.20 points** on average.

### Pre-Release Model

The pre-release model uses only information available during production (Director, Genre, Runtime, Certificate). Accuracy is lower by design — the two strongest predictors (`Meta_score` at 28% importance and `Log_Votes` at 10%) do not exist before a film releases. This is consistent with findings in academic literature (Lash & Zhao, 2015), where pre-release prediction is considered an open hard problem even with budget and studio data.

---

## Key Findings

### Feature Importance — Gradient Boosting

```
Director_Avg_Rating  ████████████████████████████████████████  38%
Meta_score           ████████████████████████████             28%
Genre_Avg_Rating     █████████████████                        17%
Log_Votes            ██████████                               10%
Runtime              ████                                      4%
Log_Gross            ███                                       3%
```

- **Director reputation (38%)** is the single most powerful predictor — more than critic scores, more than box office. A director's track record strongly forecasts audience reception.
- **Critic consensus (28%)** — professional critics and audiences agree more than commonly assumed.
- **Genre context (17%)** — a 7.5 rating in a Western (average: 7.7) means something different than a 7.5 in a Horror film (average: 6.5).
- **Box office (3%)** — commercial success is a poor proxy for quality. Marketing spend ≠ audience rating.

### EDA Highlights

- Average IMDb rating in dataset: **6.75** (std: 0.86, range: 4.6–9.3)
- Strongest raw correlation with rating: `Log_Gross` at **0.80**
- Top-rated genre by average: **Western (7.70)**, followed by Film-Noir (7.51)
- Most common genre: Comedy / Drama
- Average runtime: **111 minutes** | Median votes: **33,912** | Max votes: **2.75M**

---

## Limitations

1. **IMDB voter self-selection bias** — ratings skew toward popular English-language films; not a random sample of global opinion.
2. **Survivorship bias in votes/gross** — older films have had more time to accumulate votes; grosses are not inflation-adjusted.
3. **No NLP / sentiment analysis** — review text and social media signals (proven strong predictors in literature) are not used.
4. **Director_Avg_Rating leakage** — computed from all films in the dataset including the target film. Production-grade implementation would compute using only films released before each target film's year.
5. **No budget data** — production budget is a known predictor (Lash & Zhao, 2015) but not available in these datasets.
6. **Pre-release model limitation** — inherently less accurate because the two strongest post-release features do not exist pre-release. This is a known open problem in the field, not a modelling error.

---

## Installation & Usage

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook

```bash
git clone https://github.com/YOUR_USERNAME/imdb-rating-prediction.git
cd imdb-rating-prediction
jupyter notebook EDA_IMDB_Rating_new1.ipynb
```

### Use the Prediction Function

```python
# After running the notebook, use the trained model directly:

predicted_rating = predict_imdb_rating(
    meta_score=80,
    no_of_votes=500000,
    runtime=140,
    gross=100_000_000
)

print(f"Predicted IMDb Rating: {predicted_rating}")
# → Predicted IMDb Rating: 7.84
```

### Data Files Required

Place both CSV files in the root directory before running:

```
imdb_top_1000.csv
Top_10000_Movies_IMDb.csv
```

---

## Project Structure

```
imdb-rating-prediction/
│
├── EDA_IMDB_Rating_new1.ipynb   # Main notebook — full pipeline
├── README.md                     # This file
│
├── data/
│   ├── imdb_top_1000.csv         # IMDb Top 1,000 dataset
│   └── Top_10000_Movies_IMDb.csv # IMDb Top 10,000 dataset
│
└── outputs/
    ├── correlation_heatmap.png
    ├── rating_distribution.png
    ├── feature_importance.png
    └── model_comparison.png
```

---

## References

- Bristi, W. Z. et al. (2019). *Predicting IMDb Rating of Movies by Machine Learning Techniques.* IEEE ICCES.
- Lash, M. T. & Zhao, K. (2015). *Early Predictions of Movie Success: the Who, What, and When of Profitability.* Journal of Management Information Systems, 33(3).
- Gomes, J. et al. (2022). *Predicting IMDb Rating of TV Series with Deep Learning.* arXiv:2208.13302.
- Gupta, V. et al. (2023). *Predicting attributes based movie success through ensemble machine learning.* Multimedia Tools and Applications, 82(7).

---
