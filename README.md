# 🏘️ Birmingham House Price Predictor (2024-2026)

This project is a comprehensive Machine Learning pipeline designed to predict property prices in Birmingham using real-world **HM Land Registry** data. It demonstrates a complete data science workflow, focusing on **Iterative Model Optimization** and **Diagnostic-Driven Feature Engineering**.

## ⚖️ Data Attribution & License
**Contains HM Land Registry data © Crown copyright and database right 2021. This data is licensed under the Open Government Licence v3.0.**

* **Permitted Use**: This data is used for non-commercial purposes to display residential property price information.
* **Address Data**: Contains address data processed against Ordnance Survey’s AddressBase Premium and Royal Mail’s PAF® database.
* **Compliance**: Use of this data adheres to the terms of the Open Government Licence (OGL) v3.0.

---

## 🌟 Project Highlights
- **Real-World Scale**: Processed over 100,000 records from the UK Price Paid Dataset.
- **Target Encoding**: Developed a custom `Area_Type_Avg` feature to capture the critical interaction between location and property type.
- **Explainable AI**: Utilized Feature Importance analysis to identify and resolve model bias.

---

## 📈 Model Evolution: The Journey to $R^2 = 0.46$

The core value of this project is the documented transition from a failed baseline to a logical, signal-driven model.

### 1. Version 1: The Raw Baseline (Failed)
* **Status**: Significant Underperformance ($R^2 < 0$).
* **Diagnosis**: The model failed because it was overwhelmed by **extreme outliers** (luxury properties over £30M) and a heavily skewed price distribution.
* **Lesson Learned**: Real-world data requires aggressive cleaning and distribution normalization before modeling.

### 2. Version 2: The Overfitting Trap
* **Observation**: After filtering properties $\le$ £1M and applying log-transformation, the $R^2$ improved to 0.38, but **`Month`** unexpectedly became the top predictor.
* **Diagnosis**: The model was "memorizing" seasonal noise from 2024 instead of learning underlying property value.
* **Action**: Identified a logical flaw where the model relied on transaction timing rather than economic substance.

### 3. Version 3.1: Optimized Logic (Final Mainstream)
* **Improvements**:
    * **Composite Target Encoding**: Introduced `Area_Type_Avg` (Postcode + Property Type) to act as a stronger proxy for house size.
    * **Noise Reduction**: Completely removed the `Month` variable to force the model to prioritize stable geographic features.
    * **Complexity Control**: Restricted `max_depth` to 10 to ensure the model generalizes well on unseen 2025 data.
* **Result**: Successfully boosted **$R^2$ to 0.4624**, aligning the model with real-world housing market logic.

<p align="center">
  <img src="./feature_importance_v3_1.png" width="80%" alt="V3.1 Feature Importance">
  <br>
  <em>Figure: V3.1 Importance ranking showing the dominance of the Area-Type composite signal.</em>
</p>

---

## 🛠️ Technical Implementation
* **Data Cleaning**: Focused on the mainstream market by filtering properties $\le$ £1,000,000.
* **Mathematical Scaling**: Applied **Log-transformation** (`np.log1p`) to the price variable to normalize skewed distributions.
* **Robustness**: Used `.copy()` during data filtering to ensure data integrity and avoid `SettingWithCopy` warnings in Pandas.
* **Tech Stack**: Python (Pandas, Scikit-learn, Seaborn, Matplotlib).

---

## 🚀 Future Work
* **EPC Data Integration**: The current bottleneck is the lack of physical dimensions. The next planned step is to merge **Energy Performance Certificate (EPC)** data to include **Total Floor Area**.
* **Hyperparameter Tuning**: Implementing `GridSearchCV` to further refine parameters once physical features are added.

---