# British Airways - Task 2 Summary
## Predicting Customer Buying Behaviour

### 1. Executive Summary
- **Objective**: Build predictive model to identify customers likely to complete bookings
- **Best Model**: Random Forest Classifier
- **Key Metric**: ROC-AUC = 0.742

### 2. Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.681 |
| Precision | 0.702 |
| Recall | 0.638 |
| F1-Score | 0.669 |
| ROC-AUC | 0.742 |

### 3. Key Drivers of Booking Completion
1. **Purchase Lead Time** (Most Important)
2. **Flight Duration**
3. **Route Complexity**
4. **Number of Add-ons**
5. **Booking Origin**

### 4. Business Recommendations
1. **Early Engagement**: Target customers 30-90 days before travel
2. **Personalized Offers**: Bundle add-ons based on route and flight duration
3. **Segment Targeting**: Focus on customers with high purchase probability
4. **Real-time Personalization**: Adapt offers based on booking behavior
5. **Marketing Optimization**: Allocate resources to high-value customer segments

### 5. Next Steps
- Implement model in production environment
- Monitor model performance regularly
- Collect more granular data for improved predictions
- A/B test personalized recommendations
- Expand model to include real-time behavioral data

### 6. Visualizations
[Include the model evaluation charts from the notebook]