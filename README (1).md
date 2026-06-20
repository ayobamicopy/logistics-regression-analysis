# ✈️ Airline Customer Satisfaction Prediction - Logistic Regression Classifier

## 📌 Project Overview
This repository contains an end-to-end Machine Learning classification engine focused on predicting airline passenger satisfaction using survey parameters from **129,880 individual consumers**. Leveraging **Scikit-learn's Binomial Logistic Regression Framework**, this analysis isolates the structural features impacting passenger experiences, charts classification decision trade-offs via detailed metric reviews, and maps regression coefficients into clear executive recommendations to enhance corporate passenger retention.

---

## 📊 Dataset Properties & Preprocessing
The underlying matrix comprises 22 features tracking customer metrics and specific service rating indices on an ordinal 0-5 configuration.

### Data Engineering Steps:
* **Null Values Resolution:** Removed rows with missing values (`.dropna()`) within the continuous runtime parameters to guarantee a perfectly clean analysis.
* **Target Schema Conversion:** Mapped target strings (`satisfaction`) directly into binary classification variables (`satisfied` = 1, `dissatisfied` = 0).
* **Categorical One-Hot Encoding:** Applied one-hot transformations via `pd.get_dummies(drop_first=True)` to convert multi-class features (`Class`, `Type of Travel`, `Customer Type`) into structured numerical parameters.
* **Standard Scaling Normalization:** Executed a standard normalization scaler across train and test partitions to allow coefficients to be compared directly as clear metrics of feature importance.

---

## ⚙️ Model Setup & Training
* **Algorithm:** Binomial Logistic Regression via Scikit-learn (`LogisticRegression`).
* **Validation Strategy:** Stratified 80/20 train-test division to preserve identical target ratios across partitions.
* **Convergence Threshold:** Extended iteration metrics to guarantee complete mathematical optimization.

---

## 📈 Classification Model Performance Metrics

The predictive engine displays stable and balanced metric parameters across all validation categories, ensuring absolute robustness under structural changes:

* **Overall Classification Accuracy:** `82.53%`
* **Precision Profile (Positive Predictive Value):** `83.92%` (Minimizes false positive classifications).
* **Recall / Sensitivity Matrix:** `84.21%` (Successfully captures 84.21% of all truly satisfied travelers).
* **Balanced F1-Score:** `84.07%` (Confirms optimal baseline equilibrium between Precision and Recall boundaries).
* **Area Under the Curve (AUC-ROC):** `0.9026` (Exceptional discrimination strength between satisfaction profiles).

### 🧮 True Label Validation Matrix:
| | Predicted Dissatisfied | Predicted Satisfied |
|---|---|---|
| **Actual Dissatisfied** | **9,432** (True Negatives) | **2,289** (False Positives) |
| **Actual Satisfied** | **2,241** (False Negatives) | **11,936** (True Positives) |

---

## 🎯 Key Strategic Executive Takeaways

### Foundational Delight Enhancers:
1. **Inflight Entertainment (+0.9730):** The single most powerful positive driver of positive reviews. Content variety and modern screens yield maximum returns. This is backed explicitly by exploratory visualizations showing an exponential surge in satisfaction probability as ratings move from moderate to excellent tiers.
2. **Seat Comfort (+0.4096) & On-board Service (+0.3990):** Core physical amenities and crew attentiveness remain primary pillars of customer retention.

### Identified Service Pain-Points:
* **Disloyal Passenger Segments (-0.7319) & Economy Placements (-0.3599):** Non-member travelers and lower ticket class passengers carry the highest baseline risk of filing negative survey responses.

### Operational Adjustments:
* **Media Infrastructure Overhauls:** Route capital investment into high-leverage digital assets like movie catalogs and updated screens before updating physical terminal elements, as entertainment drives satisfaction more effectively than any other operational feature.
* **Proactive Economy Buffers:** Implement automated check-in notifications or priority terminal queues for Economy class passengers to insulate against the systemic satisfaction baseline drop.
