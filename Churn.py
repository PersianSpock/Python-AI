import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import tensorflow as tf
import autokeras as ak
import keras
import imblearn
import lightgbm as lgb
import catboost as cb


from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV,KFold,train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from category_encoders import CatBoostEncoder, TargetEncoder
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve,classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif
from sklearn.feature_selection import f_classif
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, RFE
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from keras import Sequential, regularizers
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation
from tensorflow.keras.utils import to_categorical
from lightgbm import LGBMClassifier


#Reading data
df = pd.read_csv('/content/drive/MyDrive/WA_Fn-UseC_-Telco-Customer-Churn.csv')
pd.set_option('display.max_colwidth',500)
pd.set_option('display.max_columns',100)
print(df.head(10))

print(df.info(verbose=1))

#Checking for duplicates
print(df['customerID'].duplicated().any())

#Features Engineering 
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
print(df.isnull().sum())

replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
for i in replace_cols : 
    df[i]  = df[i].replace({'No internet service' : 'No'})

le = LabelEncoder()
df1 = df.copy(deep = True)
text_data_features = [i for i in list(df.columns) if i not in list(df.describe().columns)]
for i in text_data_features :
    df1[i] = le.fit_transform(df1[i])

print(df1.describe())

#Removing missing values 
df1.dropna(inplace = True)
#Remove customer IDs from the data set
df1 = df1.iloc[:,1:]
#Convertin the predictor variable in a binary numeric variable
df1['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df1['Churn'].replace(to_replace='No',  value=0, inplace=True)
#Converting all the categorical variables into dummy variables
df_dummies = pd.get_dummies(df1)
print(df_dummies.head())

# We will use the data frame where we had created dummy variables
y = df_dummies['Churn'].values
X = df_dummies.drop(columns = ['Churn'])

# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

over = SMOTE(sampling_strategy = 1)
X, y = over.fit_resample(X, y)

#Creating train/test datasets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Checking Feautures 
col = list(df1.columns)
categorical_features = []
numerical_features = []
for i in col:
    if len(df[i].unique()) > 6:
        numerical_features.append(i)
    else:
        categorical_features.append(i)

print('Categorical Features :',*categorical_features)
print('Numerical Features :',*numerical_features)

#Checking Correlations
def cor_categorical(col):
    return df1.groupby(col)['Churn'].value_counts(normalize=True).unstack()[1].sort_values(ascending=False)    

print('''
Categorical features correlation with predictor
-----------------------------------------------''')
print(cor_categorical('gender'))
print('-'*47)
print(cor_categorical('Partner'))
print('-'*47)
print(cor_categorical('Dependents'))
print('-'*47)
print(cor_categorical('PhoneService'))
print('-'*47)
print(cor_categorical('MultipleLines'))
print('-'*47)
print(cor_categorical('InternetService'))
print('-'*47)
print(cor_categorical('OnlineSecurity'))
print('-'*47)
print(cor_categorical('OnlineBackup'))
print('-'*47)
print(cor_categorical('DeviceProtection'))
print('-'*47)
print(cor_categorical('TechSupport'))
print('-'*47)
print(cor_categorical('StreamingTV'))
print('-'*47)
print(cor_categorical('StreamingMovies'))
print('-'*47)
print(cor_categorical('Contract'))
print('-'*47)
print(cor_categorical('PaperlessBilling'))
print('-'*47)
print(cor_categorical('PaymentMethod'))


#Visualizing and EDA

colors = ['#E94B3C','#2D2926']
plt.figure(figsize = (20,5))
sns.heatmap(df1.corr(),cmap = colors,annot = True);

corr = df1.corrwith(df1['Churn']).sort_values(ascending = False).to_frame()
corr.columns = ['Correlations']
plt.subplots(figsize = (5,5))
sns.heatmap(corr,annot = True,cmap = colors,linewidths = 0.4,linecolor = 'black');
plt.title('Correlation w.r.t Outcome');

features = df1.loc[:,categorical_features]
target = df1.loc[:,'Churn']

best_features = SelectKBest(score_func = chi2,k = 'all')
fit = best_features.fit(features,target)

featureScores = pd.DataFrame(data = fit.scores_,index = list(features.columns),columns = ['Chi Squared Score']) 

plt.subplots(figsize = (5,5))
sns.heatmap(featureScores.sort_values(ascending = False,by = 'Chi Squared Score'),annot = True,cmap = colors,linewidths = 0.4,linecolor = 'black',fmt = '.2f');
plt.title('Selection of Categorical Features');

features = df1.loc[:,numerical_features]
target = df1.loc[:,'Churn']

best_features = SelectKBest(score_func = f_classif,k = 'all')
fit = best_features.fit(features,target)

featureScores = pd.DataFrame(data = fit.scores_,index = list(features.columns),columns = ['ANOVA Score']) 

plt.subplots(figsize = (5,5))
sns.heatmap(featureScores.sort_values(ascending = False,by = 'ANOVA Score'),annot = True,cmap = colors,linewidths = 0.4,linecolor = 'black',fmt = '.2f');
plt.title('Selection of Numerical Features');


#Clustering Features

print('features_train:',x_train.shape), print('target_train: ',y_train.shape)
print('features_test: ',x_test.shape), print('target_test: ',y_test.shape)

df_clusters = df1.drop('Churn',axis=1)
distortion = []
K = range(1, 8)
for k in K:
    model = KMeans(n_clusters=k, random_state=12345)
    model.fit(df_clusters)
    distortion.append(model.inertia_)

plt.figure(figsize=(12, 8))
plt.plot(K, distortion, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Objective function value')
plt.show()


#Prediction models/Classifications

def model(classifier,x_train,y_train,x_test,y_test):
    
    m = classifier.fit(x_train,y_train)
    prediction = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
    print("Cross Validation Score : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc').mean()))
    print("ROC_AUC Score : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))
    RocCurveDisplay.from_estimator(classifier, x_test,y_test)
    plt.title('ROC_AUC_Plot')
    plt.show()
    return m

def model_evaluation(classifier,x_test,y_test):
    
    # Confusion Matrix
    cm = confusion_matrix(y_test,classifier.predict(x_test))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
    
    # Classification Report
    print(classification_report(y_test,classifier.predict(x_test)))


classifier_xgb = XGBClassifier()
model_1 = model(classifier_xgb,x_train,y_train,x_test,y_test)
model_evaluation(classifier_xgb,x_test,y_test)

classifier_lr = LogisticRegression()
model_2 = model(classifier_lr,x_train,y_train,x_test,y_test)
model_evaluation(classifier_lr,x_test,y_test)

classifier_lgbm = LGBMClassifier(learning_rate= 0.01,max_depth = 3,n_estimators = 1000)
model_3 = model(classifier_lgbm,x_train,y_train,x_test,y_test)
model_evaluation(classifier_lgbm,x_test,y_test)

classifier_rf = RandomForestClassifier(max_depth = 4,random_state = 0)
model_4 = model(classifier_rf,x_train,y_train,x_test,y_test)
model_evaluation(classifier_rf,x_test,y_test)

classifier_dt = DecisionTreeClassifier(random_state = 1000,max_depth = 4,min_samples_leaf = 1)
model_5 = model(classifier_dt,x_train,y_train,x_test,y_test)
model_evaluation(classifier_dt,x_test,y_test)

classifier_gbm = GradientBoostingClassifier()
model_6 = model(classifier_gbm,x_train,y_train,x_test,y_test)
model_evaluation(classifier_gbm,x_test,y_test)

classifier_ada = AdaBoostClassifier()
model_7 = model(classifier_ada,x_train,y_train,x_test,y_test)
model_evaluation(classifier_ada,x_test,y_test)

classifier_cc = CatBoostClassifier(silent=True)
model_8 = model(classifier_cc,x_train,y_train,x_test,y_test)
model_evaluation(classifier_cc,x_test,y_test)

#Creating Stack
stack = StackingClassifier(estimators = [('classifier_xgb',classifier_xgb),
                                         ('classifier_lgbm',classifier_lgbm),
                                         ('classifier_rf',classifier_rf),
                                         ('classifier_dt',classifier_dt),
                                         ('classifier_gbm', classifier_gbm),
                                         ('classifier_ada', classifier_ada),
                                         ('classifier_cc', classifier_cc),
                                         ('classifier_lr', classifier_lr)],
                           final_estimator = classifier_cc)

model_9 = model(stack,x_train,y_train,x_test,y_test)
model_evaluation(stack,x_test,y_test)

#StructuredDataClassifier 
clf = ak.StructuredDataClassifier(max_trials=10,seed=1234)
# Feed the structured data classifier with training data.
clf.fit(x_train, y_train, epochs=50, validation_split=0.2)
# Predict with the best model.
predicted_y = clf.predict(x_test)
# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))

model_11 = clf.export_model()
print(model_11.summary())

# predictions from best neural network from AutoKeras
predicted = clf.predict(x_test)
print("classification report for neural network")
print(classification_report(predicted,y_test))


#ANN With Tensorflow 

def upsample(features, target, repeat):
    '''
    Returns upsampeled balanced tuple of 2 data sets: features and target
    Keyword arguments:
        features - features data set
        target - loyal labels data set
        repeat - integer, the ratio to multiply the minority lable
    '''
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = np.concatenate([features_ones] + [features_zeros] * repeat)
    target_upsampled = np.concatenate([target_ones] + [target_zeros] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345
    )

    return features_upsampled, target_upsampled


features_upsampled, target_upsampled = upsample(
    x_train, y_train, 3
)

def downsample(features, target, fraction):
    '''
    Returns downsampeled balanced tuple of 2 data sets: features and target
    Keyword arguments:
        features - features data set
        target - loyal labels data set
        fraction - float, the ratio to multiply the majoority lable
    '''
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    target_ones = pd.DataFrame(target_ones, columns = ["churn"])
    target_zeros = pd.DataFrame(target_zeros, columns = ["churn"])

    features_downsampled = np.concatenate(
        [features_ones.sample(frac=fraction, random_state=12345)]
        + [features_zeros]
    )
    target_downsampled = np.concatenate(
        [target_ones.sample(frac=fraction, random_state=12345)]
        + [target_zeros]
    )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )

    return features_downsampled, target_downsampled


features_downsampled, target_downsampled = downsample(
    x_train, y_train, 0.36
)

#Checking for the best model
my_optimizers = {"Mini-batch GD":tf.keras.optimizers.SGD(learning_rate = 0.1, momentum = 0.0),
                 "Momentum GD":tf.keras.optimizers.SGD(learning_rate = 0.1, momentum = 0.9),
                 "RMS Prop":tf.keras.optimizers.RMSprop(learning_rate = 0.1, rho = 0.9),
                 "Adam":tf.keras.optimizers.Adam(learning_rate = 0.1, beta_1 = 0.9, beta_2 = 0.999)
    }

histories = {}
for optimizer_name, optimizer in my_optimizers.items():
    # Define a neural network

    my_network = tf.keras.Sequential([
                                      tf.keras.layers.Dense(8, activation="relu",input_dim=(x_train.shape[1])),
                                      tf.keras.layers.Dense(4, activation="relu"),
                                      tf.keras.layers.Dense(1, activation="sigmoid")
                                    ])
    # Compile the model
    my_network.compile(optimizer=optimizer,
                       loss='binary_crossentropy', # since labels are more than 2 and not one-hot-encoded
                       metrics=[tf.keras.metrics.AUC()])
    print("*********")
    print(optimizer_name)
    print("downsampled")
    history = my_network.fit(features_downsampled, target_downsampled, epochs=20)
    print("upsampled")
    history = my_network.fit(features_upsampled, target_upsampled, epochs=20)


#The Best model Compilation 
model_12 = tf.keras.Sequential([tf.keras.layers.Dense(8, activation="relu",input_dim=(x_train.shape[1])),
                                tf.keras.layers.Dense(4, activation="relu"),
                                tf.keras.layers.Dense(1, activation="sigmoid")
                                    ])
# Compile the model
model_12.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = 0.1, momentum = 0.0),
                   loss='binary_crossentropy',
                    metrics=[tf.keras.metrics.AUC()])

print(history = model_12.fit(features_upsampled, target_upsampled, epochs=20))
print(roc_auc = model_12.evaluate(x_test, y_test))


#Hyperparameters Finetuning for best Classifiers 

#CatBoost
parameters = {'depth'         : [4,5,6,7,8,9, 10],
                 'learning_rate' : [0.1,0.2,0.3,0.4],
                  'iterations'    : [10, 20,30,40,50,60,70,80,90, 100]
                 }

import joblib
# Performing GridSearchCV
CBC = CatBoostClassifier()
Grid_CBC = GridSearchCV(estimator=CBC, param_grid = parameters, cv = 2, n_jobs=-1)
gcv_ccl_fit = Grid_CBC.fit(x_train, y_train)

# Storing CV result
#cbcl_df= pd.DataFrame(gcv_ccl_fit.cv_results_)
#joblib.dump(cbcl_df, '/content/drive/MyDrive/colab_data/xgbcl_df.pkl')
print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",Grid_CBC.best_estimator_)
print("\n The best score across ALL searched params:\n",Grid_CBC.best_score_)
print("\n The best parameters across ALL searched params:\n",Grid_CBC.best_params_)

# Using CatBoost Classifier
ccl = CatBoostClassifier(depth= 9, iterations =  90, learning_rate =  0.4)
ccl.fit(x_train, y_train)
y_train_pred= ccl.predict(x_train)
y_test_pred= ccl.predict(x_test)
print('Accuracy on trining data:' ,accuracy_score(y_train, y_train_pred))
print('Accuracy on testing data:' ,accuracy_score(y_test, y_test_pred))


print('Accuracy on testing data: \n' ,classification_report(y_train, y_train_pred))

# Classification report on testing dataset
ccl_final_cl_report_test= classification_report(y_test, y_test_pred)
print('Accuracy on testing data: \n' ,ccl_final_cl_report_test)

# XGB
param_grid= {'n_estimators': [100, 200, 300, 500],
        'gamma': [0.5, 0.7, 1],
        'subsample': [0.6,0.9, 1],
        'colsample_bytree': [0.6, 0.9, 1],
        'max_depth': [4, 6, 8, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.5]
        }

# Performing GridSearchCV
xgbcl= XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', random_state= 42)
gcv_xgbcl= GridSearchCV(estimator= xgbcl, param_grid= param_grid, cv= 3, verbose=3, n_jobs= -1, scoring= 'recall', return_train_score= True)
gcv_xgbcl_fit= gcv_xgbcl.fit(x_train, y_train)

# Storing CV result
#xgbcl_df= pd.DataFrame(gcv_xgbcl_fit.cv_results_)
#joblib.dump(xgbcl_df, '/content/drive/MyDrive/colab_data/xgbcl_df.pkl')

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",gcv_xgbcl.best_estimator_)
print("\n The best score across ALL searched params:\n",gcv_xgbcl.best_score_)
print("\n The best parameters across ALL searched params:\n",gcv_xgbcl.best_params_)

# Using XGBoost Classifier
xgbcl= XGBClassifier(booster='gbtree',colsample_bytree= 1, subsample= 1, gamma= 1,  learning_rate= 0.5, max_depth= 9, n_estimators= 500, n_jobs= -1,
              random_state=42, verbosity=1)
xgbcl.fit(x_train, y_train)
y_train_pred= xgbcl.predict(x_train)
y_test_pred= xgbcl.predict(x_test)
print('Accuracy on trining data:' ,accuracy_score(y_train, y_train_pred))
print('Accuracy on testing data:' ,accuracy_score(y_test, y_test_pred))

# Classification report on training dataset
print('Accuracy on testing data: \n' ,classification_report(y_train, y_train_pred))

# Classification report on testing dataset
xgbc_final_cl_report_test= classification_report(y_test, y_test_pred)
print('Accuracy on testing data: \n' ,xgbc_final_cl_report_test)


#Saving the best model

# serialize model to YAML
model_yaml = model_12.to_json()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model_12.save_weights("fina_churn_model.h5")
print("Saved model to disk")




