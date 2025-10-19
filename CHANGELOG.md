# CHANGELOG
## v0.1
- Baseline: StandardScaler + LinearRegression
- Metric: RMSE = 53.85345, Risk Quantile = 0.75, Threshold = 214, Precision (High risk) = 1.0, Recall = 0.28

## v0.2
- Improvmed Model: RandomForestRegressor (adds non-linear capacity and robustness without a need to scale. 
- Metric: RMSE = 54.82 , slightly higher than v0.1 but still compareable accuracy. 
- Risk Quantile = 0.75,
- Threshold = 214
- Precision (High risk) = 1.0
- Recall = 0.33
- Precision: 0.87

Reflection: 
Accuracy remains compareble to the basline. And triage behavior moves toward higher sensitivity at a slightly cost in precision but considered preferable since missing high-risk patiens is costly. 
