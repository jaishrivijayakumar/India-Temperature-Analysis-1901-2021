# Seasonal and Annual Mean Temperature Analysis (1901–2021)

A machine learning project to analyze and predict seasonal mean temperatures in India across 121 years using regression models trained on historical climate data.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Model Results](#model-results)
- [Author](#author)

---

## Project Overview

This project builds an end-to-end ML regression pipeline to predict the March–May (MAR-MAY) seasonal mean temperature using annual and other seasonal temperature records spanning 1901 to 2021. The dataset contains 121 yearly records with mean temperature values across four seasons and an annual average.

The target variable is `MAR-MAY` — the mean temperature for the pre-monsoon spring season. The pipeline covers the full ML workflow: data loading, exploratory data analysis with boxplots and bar charts, missing value handling, outlier treatment, feature scaling, feature selection, model training, and R² score evaluation across 7 regression models.

---

## Repository Structure

```
Seasonal-Temperature-Analysis/
├── Seasonal_and_Annual_Mean_Temperature_Series_for_the_period_1901_to_2021.ipynb    # End-to-end ML pipeline notebook
└── Seasonal_and_Annual_Mean_Temperature_Series_for_the_period_1901_to_2021.csv     # Dataset (121 yearly records)
```

---

## Dataset

**File:** `Seasonal_and_Annual_Mean_Temperature_Series_for_the_period_1901_to_2021.csv`  
**Source:** Hugging Face  
**Total Records:** 121 (one per year from 1901 to 2021)  
**Target Variable:** `MAR-MAY` — Mean temperature for the March–May season

| Feature | Type | Description |
|---|---|---|
| `YEAR` | Integer | Year of observation (1901–2021) |
| `ANNUAL` | Float | Annual mean temperature |
| `JAN-FEB` | Float | Mean temperature for January–February (Winter) |
| `MAR-MAY` | Float | Mean temperature for March–May (Pre-monsoon) — Target |
| `JUN-SEP` | Float | Mean temperature for June–September (Monsoon) |
| `OCT-DEC` | Float | Mean temperature for October–December (Post-monsoon) |

---

## ML Pipeline

### 1. Data Loading and Exploration

- Loaded dataset using `pandas`
- Inspected shape, data types, and missing value counts using `df.info()` and `df.describe()`
- Visualized distributions of all columns using **boxplots** (`matplotlib`)
- Plotted average temperature per season using a **bar chart**

### 2. Missing Value Handling

- **Numerical columns** — imputed using mean via `SimpleImputer(strategy='mean')`
- **Categorical columns** — imputed using mode via `SimpleImputer(strategy='most_frequent')` (none present in this dataset)

### 3. Categorical Encoding

- Applied **Label Encoding** for any binary categorical columns via `LabelEncoder`
- No categorical columns were present in this dataset — step was included as part of the standard pipeline

### 4. Outlier Treatment

- Used **IQR (Interquartile Range)** method for all numerical columns
- Capped values below `Q1 - 1.5 * IQR` to the lower bound
- Capped values above `Q3 + 1.5 * IQR` to the upper bound

### 5. Feature Selection

- Applied `SelectKBest` with `f_regression` to identify the most relevant features for predicting `MAR-MAY`
- Visualized feature correlations using a **heatmap** (`seaborn`)
- Printed correlation values sorted by `MAR-MAY`

### 6. Feature Scaling

- Applied `StandardScaler` to normalize all feature values before model training

### 7. Train-Test Split

| Parameter | Value |
|---|---|
| Test size | 30% |
| Train size | 70% |
| Random state | 100 |

---

## Model Results

All 7 regression models were trained on the same preprocessed dataset and evaluated using **R² Score**.

| # | Model | R² Score |
|---|---|---|
| 1 | **Linear Regression** | **0.9988** |
| 2 | Ridge Regression | 0.9839 |
| 3 | Support Vector Regression | 0.7593 |
| 4 | Random Forest | 0.6103 |
| 5 | Gradient Boosting | 0.5893 |
| 6 | K-Nearest Neighbors | 0.3956 |
| 7 | Decision Tree | 0.0844 |

**Best performing model: Linear Regression — R² Score of 0.9988**

> The high R² score of Linear Regression indicates a strong linear relationship between the seasonal temperature features and the MAR-MAY target variable, which is expected given the consistent and gradual nature of climate data over time.

---

## Author

**Jaishri Vijayakumar**  
B.Sc. Data Science | PSGR Krishnammal College for Women, Coimbatore  
LinkedIn: linkedin.com/in/jaishri-vijayakumar  
GitHub: github.com/jaishrivijayakumar
