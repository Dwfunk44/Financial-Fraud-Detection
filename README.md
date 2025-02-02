Fraud Detection Model Comparison: Logistic Regression vs. Random Forest

Project Overview

This project aims to build and compare two machine learning models (Logistic Regression and Random Forest) for detecting fraudulent transactions in a financial dataset. The goal is to classify transactions as either fraudulent or non-fraudulent, utilizing features such as transaction amounts, account balances, and transaction types. The dataset is imbalanced, with fraud transactions being much less frequent than non-fraudulent ones, which makes it essential to focus on precision, recall, and F1-score in evaluating model performance.

Dataset

The dataset used in this project is a simulated financial transaction dataset with the following key features:

amount: The amount of the transaction.
oldbalanceOrg: The balance in the origin account before the transaction.
newbalanceOrig: The balance in the origin account after the transaction.
oldbalanceDest: The balance in the destination account before the transaction.
newbalanceDest: The balance in the destination account after the transaction.
amount_percentage: The transaction amount as a percentage of the origin account balance.
balance_diff: The difference in the balance of the destination account before and after the transaction.
type: The type of transaction (e.g., cash-in, cash-out, transfer, etc.).
isFraud: The target variable indicating whether the transaction is fraudulent (1) or not (0).
Approach

Data Preprocessing:
The dataset was preprocessed by handling missing values, encoding categorical variables, and scaling numerical features to prepare the data for model training.
The type column (categorical) was one-hot encoded to convert it into a numerical format.
The features were scaled using StandardScaler to ensure that they are on a similar scale, which is particularly important for models like Logistic Regression.
Model Training:
Two classification models were chosen: Logistic Regression and Random Forest.
Logistic Regression is a linear model that assumes a linear relationship between the input features and the log-odds of the target variable.
Random Forest is an ensemble method that builds multiple decision trees and aggregates their predictions to improve performance and reduce overfitting.
Both models were trained on a sample of 10% of the dataset (to speed up the process), and their performance was evaluated using an 80/20 train-test split.
Model Evaluation:
The performance of the models was assessed using multiple metrics, including:
Accuracy: The percentage of correctly classified transactions.
Confusion Matrix: The matrix showing true positives, true negatives, false positives, and false negatives.
Precision: Of all transactions predicted as fraud, how many were truly fraudulent.
Recall: Of all actual fraudulent transactions, how many were correctly identified.
F1-Score: The harmonic mean of precision and recall, providing a balance between the two.
The ROC curve and AUC (Area Under the Curve) were also analyzed to visualize the trade-off between true positive rate and false positive rate.
Results

Accuracy:
Logistic Regression achieved an accuracy of 99.91%, while Random Forest performed slightly better with an accuracy of 99.96%.
Confusion Matrix:
Both models showed a high number of true negatives (correctly predicting non-fraud transactions).
Logistic Regression had more false negatives compared to Random Forest, meaning it missed more fraudulent transactions.
Random Forest, however, did a better job in identifying fraudulent transactions, showing fewer false negatives.
Precision, Recall, and F1-Score:
Logistic Regression had high precision (0.99) but lower recall (0.40), indicating it was good at identifying non-fraud transactions as non-fraud but missed many fraudulent ones.
Random Forest showed a better balance, with higher precision (0.97) and recall (0.73), indicating it was more capable of detecting fraud while keeping false positives lower.
Visualizations

Accuracy Comparison: A bar plot comparing the accuracy of both models.
Confusion Matrix: Heatmaps showing the confusion matrix for both Logistic Regression and Random Forest.
Precision, Recall, and F1-Score: A bar plot comparing the classification metrics for both models.
Conclusion

Random Forest outperformed Logistic Regression in terms of recall, meaning it was better at identifying fraud, which is critical in fraud detection tasks.
The Logistic Regression model, while achieving high precision, struggled with recall, potentially missing a significant number of fraudulent transactions.
Given the importance of recall in fraud detection (to catch as many fraudulent transactions as possible), Random Forest is the preferred model in this case.
In future iterations of the model, additional techniques such as SMOTE (Synthetic Minority Over-sampling Technique) could be used to address the class imbalance, which may help further improve model performance, especially recall.

Installation

To run the code and replicate the analysis, you'll need the following Python libraries:

pip install pandas scikit-learn matplotlib
Usage

Clone this repository.
Ensure you have the required libraries installed.
Place your engineered_transactions.csv dataset in the same directory as the script.
Run the script to train the models and generate the evaluation plots.
License

This project is licensed under the MIT License - see the LICENSE file for details.
