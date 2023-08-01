# Credit Risk Prediction
The objective of this project is to develop a robust machine learning model that can accurately predict credit risk for loan applicants based on historical data.

# File Explanation on Github
This repository consists of several files, namely :
- **notebook_Credit_Risk_Prediction.ipynb** = This file is the main notebook used to explore dataset and built model
- **inference_credit_risk_prediction.ipynb** = Notebook used for testing inference. Inferencing is done on a separate notebook to prove that the model can run on a notebook that is clean of variables
- **data_inference.csv** = Dataset used for model inference
- **low_risk.txt** = list of loan status which is included in the low risk category
- **sel_features.txt** = list of selected features
- **best_model.pkl** = TSaved model to be loaded and used later for making predictions on new data without the need to retrain the SVM.

# Brief of Summary of Project
The flow of this project, first EDA (Exploratory Data Analysis) to find out the basic picture of the dataset. Second, cleaning and preprocessing of the dataset. Third, Built Classification Models using 2 algorithms (Logistic Regression and Decision Tree ). These algorithms are tested based on their baseline/default parameters, and then cross-validation will be applied to evaluate each model based on mean and also standard deviation. Next hyperparameter tuning is carried out using the selected algorithm. The Selected model is **Decision Tree model** has been improved with Hyperparameter Tuning using GridSearch.

# Project Conclusion
1. Based on Exploratory Data Analysis (EDA) :
    - The dataset's credit_risk column shows an imbalanced class distribution, with a higher number of borrowers categorized as Low risk compared to High risk borrowers.
    - The minimum total received principal is 0.00, indicating loans that have not received any principal payments yet. The maximum total received principal is 35,000.03, suggesting significant repayments for some loans.
    - The most common loan grade is 'B', followed closely by 'C'. This suggests that the majority of borrowers fall into the 'B' and 'C' risk categories, which might indicate that the lending platform is relatively cautious in offering loans to higher-risk borrowers (grades D to G).
    - The earliest credit line available in the dataset dates back to January 1969, indicating that the lending platform serves borrowers with a wide range of credit histories, including those with a long credit history. This diversity in credit histories might contribute to the platform's ability to cater to borrowers with varying creditworthiness.
    - Borrowers with an employment length of '10+ years' have a notably higher count of 'Low-Risk' loans, indicating that longer employment experience may be associated with lower credit risk.
2. Based on Model Evaluation:
    - The Selected model is Decision Tree model has been improved with Hyperparameter Tuning using GridSearch. The classification report evaluates the performance of a model on a test set. The model achieves high accuracy (97%) in correctly classifying cases. It demonstrates strong precision, recall, and F1-scores for both high-risk (class 0.0) and low-risk (class 1.0) borrowers. However, the class imbalance may affect sensitivity to the high-risk class. While the model generalizes well, there is a need for further validation on unseen data to address potential overfitting. Interpretability and contextual understanding of predictions should be considered, and problem context may need refinement for accurate risk level representation.
3. Business Insights:
    - The analysis reveals that the average recoveries and collection recovery fees are relatively low on average. The lending platform could explore ways to enhance recovery efforts for loans that are deemed High-risk. Implementing effective collection strategies, collaborating with collection agencies, or offering more favorable payment options to struggling borrowers could help in reducing default rates and improving overall profitability.
    - Understanding the distribution of loans across different grades highlights that most borrowers are in the 'B' and 'C' risk categories. This implies that the lending platform might be more conservative in offering loans to higher-risk borrowers. To potentially increase lending opportunities and overall revenue, the platform could assess the possibility of expanding its offerings to include higher-grade loans or exploring ways to mitigate risks effectively for lower-grade loans.
4. Further Improvement:
    - Try another classification model like gradient boosting i.e AdaBoost and XGBoost