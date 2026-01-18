## 📈 Model Evolution & Performance

### Version 1: Initial Baseline
- **Approach**: Simple Random Forest on raw data.
- **Results**: $R^2 = -0.09$, MAE = £91,921.
- **Observation**: The model was heavily skewed by luxury property outliers (up to £35M), making it useless for mainstream market prediction.

### Version 2: Optimized Model
- **Improvements**: 
  - Filtered out properties > £1M to focus on the mainstream Birmingham market.
  - Applied **Log Transformation** to the target price to address the highly skewed distribution.
- **Results**: $R^2 = 0.385$, MAE = £68,733.
- **Conclusion**: Performance improved significantly. The remaining variance is likely due to missing structural features (e.g., number of bedrooms, floor area) not present in the government's Price Paid dataset.