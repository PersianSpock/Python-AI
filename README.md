# Python-AI(Project for Rayankar)


# Business problem overview
In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.

For many incumbent operators, retaining high profitable customers is the number one business goal.

To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.

Here we will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.

Customers usually do not decide to switch to another competitor instantly, but rather over a period of time (this is especially applicable to high-value customers). In churn prediction, we assume that there are three phases of customer lifecycle :

The ‘good’ phase: In this phase, the customer is happy with the service and behaves as usual.

The ‘action’ phase: The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. Also, it is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)

The ‘churn’ phase: In this phase, the customer is said to have churned. You define churn based on this phase. Also, it is important to note that at the time of prediction (i.e. the action months), this data is not available to you for prediction. Thus, after tagging churn as 1/0 based on this phase, you discard all data corresponding to this phase.

# Dataset Description
1. customerID Customer ID
2. gender Whether the customer is a male or a female
3. SeniorCitizen Whether the customer is a senior citizen or not (1, 0)
4. Partner Whether the customer has a partner or not (Yes, No)
5. Dependents Whether the customer has dependents or not (Yes, No)
6. tenure Number of months the customer has stayed with the company
7. PhoneService Whether the customer has a phone service or not (Yes, No)
8. MultipleLines Whether the customer has multiple lines or not (Yes, No, No phone service)
9. InternetService Customer’s internet service provider (DSL, Fiber optic, No)
10. OnlineSecurity Whether the customer has online security or not (Yes, No, No internet service)
11. OnlineBackup Whether the customer has online backup or not (Yes, No, No internet service)
12. DeviceProtection Whether the customer has device protection or not (Yes, No, No internet service)
13. TechSupport Whether the customer has tech support or not (Yes, No, No internet service)
14. StreamingTV Whether the customer has streaming TV or not (Yes, No, No internet service)
15. StreamingMovies Whether the customer has streaming movies or not (Yes, No, No internet service)
16. Contract The contract term of the customer (Month-to-month, One year, Two year)
17. PaperlessBilling Whether the customer has paperless billing or not (Yes, No)
18. PaymentMethod The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
19. MonthlyCharges The amount charged to the customer monthly
20. TotalCharges The total amount charged to the customer
21. Churn Whether the customer churned or not (Yes or No)

# EDA

We can see high class imbalance in data.
Most of the numeric features are right skewed. We'll take care of this during scaling by performing MinMax Scaling, then using SMOTE.
Most of the features have high correlation. As, first we want to build an interpretable model, we can't perform PCA as it'll change the actual features and Principal Components will not have any business interpretation. At the end, training dataset with PCA gave us lower auc-roc so in the final code we're not using PCA. We used Oversampling of minority class using SMOTE.

# Classification

We used XGBClassifier, LogisticRegression, LGBMClassifier, RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier, AdaBoostClassifier, CatBoostClassifier, and StackingClassifier of them all with the final_estimator as CatBoostClassifier since it had the best ROC_AUC Score. Also we used StructuredDataClassifier from autokeras. At the end the best scoring was for CatBoostClassifier with 85% followed by XGBClassifier with 84.5%. So we did Hyperparameters Tuning using GridSearch for them both. The best score after the tuning became 86% for CatBoostClassifier which is not that different form the original one.

# Neural Network Model

We created ANN using tensorflow.keras.Sequential with various optimizers and the best scoring was for: activations as 'relu' - 'relu' - 'sigmoid', Binary Cross Entropy as loss function, and Mini-batch GD as optimizer and learning rate = 0.1

We also did Upsampling and Downsmapling on dataset as well and at the end the best scoring belonged to Upsampling with 86%.

