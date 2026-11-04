# Machine Learning Course Datasets

A collection of **10 CSV datasets** for teaching and practicing Machine Learning with Python
(pandas, scikit-learn, matplotlib). Each dataset is built for a specific ML topic.

> All data is **synthetic** (computer-generated with realistic patterns). It contains no real
> people or companies, so it is free to use, share and modify for teaching.

## Datasets at a glance

| # | File | Rows | ML Task | Target column | Good for teaching |
|---|------|------|---------|---------------|-------------------|
| 1 | `salary_experience.csv` | 100 | Regression | `monthly_salary_usd` | Simple linear regression (first model) |
| 2 | `house_prices.csv` | 600 | Regression | `price_usd` | Multiple linear regression, one-hot encoding |
| 3 | `student_performance.csv` | 800 | Regression / Classification | `final_score` / `passed` | Same data, two problem types |
| 4 | `loan_approval.csv` | 1000 | Binary classification | `approved` | Logistic regression, decision trees |
| 5 | `customer_churn.csv` | 1500 | Binary classification (imbalanced) | `churn` | Precision/recall, ROC-AUC, random forest |
| 6 | `crop_recommendation.csv` | 720 | Multi-class classification | `crop` | kNN, scaling, confusion matrix |
| 7 | `mall_customers.csv` | 300 | Clustering (unsupervised) | — | K-Means, elbow method, silhouette |
| 8 | `daily_sales.csv` | 1096 | Time series | `sales_usd` | Trend, seasonality, forecasting |
| 9 | `sms_spam.csv` | 1000 | Text classification | `label` | Bag-of-words, TF-IDF, Naive Bayes |
| 10 | `employees_dirty.csv` | 412 | Data cleaning | `left_company` | Missing values, outliers, duplicates |

## Column descriptions

### 1. salary_experience.csv
| Column | Description |
|---|---|
| `years_experience` | Years of work experience (0–20) |
| `monthly_salary_usd` | Monthly salary in USD |

### 2. house_prices.csv
| Column | Description |
|---|---|
| `city` | Mogadishu, Hargeisa, Bosaso, Garowe, Kismayo, Galkayo |
| `area_sqm` | House area in square metres |
| `bedrooms`, `bathrooms` | Number of rooms |
| `age_years` | Age of the building |
| `distance_to_center_km` | Distance to the city centre |
| `has_parking` | yes / no |
| `price_usd` | **Target** – sale price in USD |

### 3. student_performance.csv
| Column | Description |
|---|---|
| `student_id` | ID (drop before training) |
| `gender` | male / female |
| `study_hours_per_day` | Average daily study hours |
| `attendance_pct` | Class attendance % |
| `previous_exam_score` | Score in previous exam (0–100) |
| `sleep_hours` | Average sleep per night |
| `internet_access`, `private_tutoring` | yes / no |
| `final_score` | **Regression target** (0–100) |
| `passed` | **Classification target** (1 if final_score ≥ 50) |

> ⚠️ When predicting `passed`, drop `final_score` – otherwise it is data leakage (a great lesson!).

### 4. loan_approval.csv
| Column | Description |
|---|---|
| `applicant_id` | ID (drop) |
| `monthly_income_usd` | Applicant's monthly income |
| `loan_amount_usd` | Requested loan amount |
| `loan_term_months` | 6, 12, 24 or 36 |
| `credit_score` | 350–850 |
| `employment_status` | employed / self_employed / unemployed |
| `dependents` | Number of dependents |
| `has_collateral` | yes / no |
| `approved` | **Target** – 1 approved, 0 rejected |

### 5. customer_churn.csv (telecom)
| Column | Description |
|---|---|
| `customer_id` | ID (drop) |
| `age` | Customer age |
| `tenure_months` | Months as a customer |
| `contract_type` | monthly / one_year / two_year |
| `monthly_charges_usd`, `total_charges_usd` | Billing |
| `monthly_data_gb` | Data usage |
| `support_calls` | Number of complaints/support calls |
| `uses_mobile_money` | yes / no |
| `churn` | **Target** – 1 left the company (~26%), 0 stayed |

### 6. crop_recommendation.csv
| Column | Description |
|---|---|
| `rainfall_mm` | Annual rainfall |
| `temperature_c` | Average temperature |
| `soil_ph` | Soil pH |
| `nitrogen` | Soil nitrogen level |
| `humidity_pct` | Humidity % |
| `crop` | **Target** – sorghum, maize, sesame, rice, banana, cowpea |

### 7. mall_customers.csv
| Column | Description |
|---|---|
| `customer_id`, `gender`, `age` | Customer info |
| `annual_income_k_usd` | Annual income in thousands USD |
| `spending_score` | 1–100 score assigned by the mall |

> Hint: plot `annual_income_k_usd` vs `spending_score` – there are 5 natural groups.

### 8. daily_sales.csv
| Column | Description |
|---|---|
| `date` | 2023-01-01 to 2025-12-31 |
| `day_of_week` | Monday … Sunday (Friday has lower sales) |
| `promotion` | 1 if a promotion ran that day |
| `temperature_c` | Daily temperature |
| `sales_usd` | **Target** – daily sales |

### 9. sms_spam.csv
| Column | Description |
|---|---|
| `label` | ham (normal, ~80%) / spam (~20%) |
| `message` | SMS text |

### 10. employees_dirty.csv
Intentionally messy dataset. Problems hidden inside:
- Missing values in `age`, `education`, `monthly_salary_usd`, `performance_rating`
- Inconsistent text: `"IT"` vs `"it "`, `"Yes"` vs `"yes"`
- Outliers/impossible values: age 150, salary −500, salary 99999
- 12 duplicated rows

## Quick start

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("datasets/salary_experience.csv")
X = df[["years_experience"]]
y = df["monthly_salary_usd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
print("R² score:", model.score(X_test, y_test))
```

Load directly from GitHub (replace `USERNAME/REPO`):
```python
url = "https://raw.githubusercontent.com/USERNAME/REPO/main/datasets/house_prices.csv"
df = pd.read_csv(url)
```

## Regenerating the data
```bash
pip install numpy pandas
python scripts/generate_datasets.py
```
The random seed is fixed (`42`), so you get the same files every time. Change the seed or
row counts in the script to create new versions (e.g. a different dataset for exams).

## License
Free to use for teaching and learning (MIT).
# ML-Datasets
