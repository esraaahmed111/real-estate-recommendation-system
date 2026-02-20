# Real Estate Recommendation System

# Overview
This project is a **machine learning-powered real estate system** designed to assist buyers in finding the most relevant properties and help sellers position their listings effectively. The system combines a **KNN-based recommendation engine** with an **ensemble regression model** for price prediction, handling both **numerical** and **categorical data**. The solution uses advanced data processing techniques to ensure accurate predictions and provide personalized property recommendations based on user preferences.
  
- **Machine Learning Models**:
  - **k-Nearest Neighbors (KNN)**: Used for property recommendations by finding similar properties based on user inputs like city, property type, and price range.
  - **HistGradientBoostingRegressor**: A gradient boosting model used for predicting property prices based on historical data, ensuring high efficiency and speed for large datasets.
  - **BaggingRegressor**: An ensemble method using multiple decision trees to improve model performance and reduce overfitting.
  - **DecisionTreeRegressor**: For modeling the relationship between property features and market prices, creating interpretable decision boundaries.
