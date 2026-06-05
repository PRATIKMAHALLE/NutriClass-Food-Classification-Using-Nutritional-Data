# NutriClass: Food Classification Using Nutritional Data

**Skills:** Data Preprocessing, Feature Engineering, Multi-class Classification, Evaluation Metrics  
**Domain:** Food and Nutrition / Machine Learning

## Problem Statement

Classify food items into categories using nutritional attributes like calories, proteins, carbs, fats, and sugar. Build a robust system for accurate food labeling and insights into category distinctions.

### Business Use Cases
- Smart dietary apps for nutritional balance recommendations.
- Health tools for diet planning.
- Food logging with auto-classification.
- Educational platforms on nutrition.
- Grocery/meal apps for category-based suggestions. 

## Approach
1. **Data Exploration:** Load dataset, check class distribution, visualize variations, assess imbalance/noise.
2. **Preprocessing:** Impute missing values, cap outliers, remove duplicates, standardize features.
3. **Feature Engineering:** PCA for dimensionality reduction, label encoding for categoricals.
4. **Models:** Compare Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost, Gradient Boosting.
5. **Evaluation:** Accuracy, Precision, Recall, F1, Confusion Matrix. 

**Target:** Food_Name (exact prediction for strict diet control). 

## Dataset
- [synthetic_food_dataset_imbalanced.csv]
- Features: Calories, Protein, Fat, Carbs, Sugar, Fiber, Sodium, Cholesterol, Glycemic_Index, Water_Content, Serving_Size, Meal_Type, Preparation_Method, Is_Vegan, Is_Gluten_Free.
