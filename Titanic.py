# -*- coding: utf-8 -*-
"""
Code to predict Titanic survival on Kaggle

https://www.kaggle.com/competitions/titanic/overview
"""

# Libraries #
import pandas as pd # Dataframes
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from math import ceil


# Import Training Data #
data = pd.read_csv('train.csv')


# Drop Unneeded Variables #
train = data.drop(['Name', 'Ticket', 'Cabin'], axis = 1)


#########################
#### Processing Data ####
#########################

# Split into X and y #
X = train.drop(['Survived', 'PassengerId'], axis = 1)
y = pd.DataFrame(train['Survived'])


# Categorical #
cat_features = (['Sex', 'Embarked'])


# Ordinal Encode #
ord_enc = OrdinalEncoder()
X[['Sex', 'Embarked']] = ord_enc.fit_transform(X[['Sex', 'Embarked']])


# KNN Imputer #
KNN_impute = KNNImputer(n_neighbors=3, weights = 'uniform')
X_2 = pd.DataFrame(KNN_impute.fit_transform(X), columns=X.columns)


# Make whole numbers for categorical features after impute #
X_2[cat_features] = X_2[cat_features].apply(lambda x: x.apply(ceil))


# Numerical List #
num_stand = (['Age', 'Fare'])


# Categorical List #
cat_stand = (['Pclass', 'SibSp', 'Parch', 'Sex', 'Embarked'])


# Categorical transformer pipeline #
cat_trans = Pipeline(steps = [
    ('encode', OneHotEncoder(sparse_output = False))])


# Numerical transformer pipeline #
num_trans = Pipeline(steps = [
    ('scale', StandardScaler())])


# Build Processor #
processor = ColumnTransformer(
    transformers=[
        ('num', num_trans, num_stand),
        ('cat', cat_trans, cat_stand)],
    remainder='passthrough')


# Process data #
temp = processor.fit_transform(X_2)


# Get categorical feature names #
enc_cat_features = list(processor.named_transformers_['cat']['encode']\
                        .get_feature_names_out())
    
    
# Concat label names #
labels = num_stand + enc_cat_features


# Make DF of processed data #
X_processed = pd.DataFrame(temp, columns=labels)
del train, temp, labels, X_2


#############################
#### Logistic Regression ####
#############################

from sklearn.linear_model import LogisticRegression
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score


# Define objective for Logistic Regression #
def objective_logistic(C, penalty):
    """"
    Objective function for bayes optimization for logistic regression.
    I allow the penalty and C value to vary for tuning
    """
    
    # Penalty variation #
    if penalty <= 0.5:
        penalty = 'l1'
    else:
        penalty = 'l2'

    # Create a logistic regression model with specified hyperparameters
    model = LogisticRegression(C=C, penalty=penalty, solver='liblinear')

    # Perform cross-validation and calculate mean accuracy
    accuracy = cross_val_score(model, X_processed, y, cv=5).mean()

    # Return accuracy score #
    return accuracy


# Define the search space
pbounds = {'C': (0.00000001, 1000),
           'penalty': [0, 1]}


# Set the optimizer #
optimizer = BayesianOptimization(f=objective_logistic,
                                 pbounds=pbounds, random_state=1)


# Call the maximizer #
optimizer.maximize(init_points=50, n_iter=200)


# Pull best info #
best_hyperparameters = optimizer.max['params']
best_accuracy = optimizer.max['target']


# Comparison Matrix #
final = pd.DataFrame(columns=['Model', 'Accuracy'])
final = final.append({'Model': 'Logistic',
                      'Accuracy': best_accuracy}, 
                      ignore_index=True)


################################
#### Support Vector Machine ####
################################

from sklearn.svm import SVC


# Define objective function for SVM #
def objective_SVM(C, gamma):

    """
    Objective function for the support vector classifier.
    I allow the C and gamma paramters to vary with the bayes optimization. 
    """
    
    # Create SVM with specified hyperparamters #
    model = SVC(C = C, gamma = gamma)

    # Perform cross-validation and calculate mean accuracy
    accuracy = cross_val_score(model, X_processed, y, cv=5).mean()

    # Return accuracy #
    return accuracy


# Define the search space #
pbounds = {'C': (0.1, 1000),
           'gamma': (0.0001, 1)}


# Set the optimizer #
optimizer = BayesianOptimization(f=objective_SVM,
                                 pbounds=pbounds,
                                 random_state=1)


# Call the maximizer #
optimizer.maximize(init_points=50, n_iter=200)


# Pull best info #
best_hyperparameters = optimizer.max['params']
best_accuracy = optimizer.max['target']


# Comparison Matrix #
final = final.append({'Model': 'SVM', 
                      'Accuracy': best_accuracy},
                      ignore_index=True)


#######################
#### Random Forest ####
#######################

from sklearn.ensemble import RandomForestClassifier

# Define objective function for Random Forest #
def objective_rf(n_estimators, max_features, max_depth, min_samples_split,
              min_samples_leaf, bootstrap, max_leaf_nodes):
    """
    This is an objective function for the bayes hyperparamter 
    tuning of the random forest. 
    The code below varies the max feature procedure and the 
    bootstrap paramter. 
    The pbounds varies the other numeric parameters.
    """

    # Vary max features procedure #
    if max_features <= 0.5:
        max_features = 'sqrt'
    else:
        max_features = 'log2'

    # Vary the bootstrap #
    if bootstrap <= 0.5:
        bootstrap = True
    else:
        bootstrap = False

    # Create random forest with specified hyperparamters #
    model = RandomForestClassifier(n_estimators=int(n_estimators),
                                   max_features=max_features,
                                   max_depth=int(max_depth),
                                   min_samples_split=int(min_samples_split),
                                   min_samples_leaf=int(min_samples_leaf),
                                   bootstrap=bootstrap,
                                   max_leaf_nodes=int(max_leaf_nodes))

    # Perform cross-validation and calculate mean accuracy
    accuracy = cross_val_score(model, X_processed, y, cv=5).mean()

    # Return accuracy #
    return accuracy


# Define the search space #
pbounds = {'n_estimators': (50, 300),
           'max_features': (0, 1),
           'max_depth': (10, 100),
           'min_samples_split': (2, 10),
           'min_samples_leaf': (1, 10),
           'bootstrap': (0, 1),
           'max_leaf_nodes': (25, 100)}


# Set the optimizer #
optimizer = BayesianOptimization(f=objective_rf, pbounds=pbounds,
                                 random_state=1)


# Call the maximizer #
optimizer.maximize(init_points=20, n_iter=200)


# Pull best info #
best_hyperparameters = optimizer.max['params']
best_accuracy = optimizer.max['target']


# Comparison Matrix #
final = final.append({'Model': 'Random Forest', 
                      'Accuracy': best_accuracy}, 
                      ignore_index=True)


##############################
#### Bagged Random Forest ####
##############################

from sklearn.ensemble import BaggingClassifier


# Define objective function for Bagged Random Forest #
def objective_bag(rf_n_estimators, rf_max_features, rf_max_depth,
                  rf_min_samples_split, rf_min_samples_leaf, rf_bootstrap,
                  rf_max_leaf_nodes, bag_n_estimators, bag_max_samples,
                  bag_max_features):
    """
    This is the objective function for the bayesian hyperparamter tuning of the
    bagged random forest model.
    The code below varies the max feature and boostrap process. 
    The numeric parameters vary with pbounds.
    """

    # Max features #
    if rf_max_features <= 0.5:
        rf_max_features = 'sqrt'
    else:
        rf_max_features = 'log2'

    # Bootstrap #
    if rf_bootstrap <= 0.5:
        rf_bootstrap = True
    else:
        rf_bootstrap = False

    # Create Random Forest with specified hyperparamters #
    rf = RandomForestClassifier(n_estimators=int(rf_n_estimators),
                                max_features=(rf_max_features),
                                max_depth=int(rf_max_depth),
                                min_samples_split=int(rf_min_samples_split),
                                min_samples_leaf=int(rf_min_samples_leaf),
                                bootstrap=rf_bootstrap,
                                max_leaf_nodes=int(rf_max_leaf_nodes))

    # Bagging Estimator #
    bag = BaggingClassifier(estimator = rf,
                            n_estimators = int(bag_n_estimators),
                            max_samples = int(bag_max_samples),
                            max_features = int(bag_max_features))

    # Perform cross-validation and calculate mean accuracy
    accuracy = cross_val_score(bag, X_processed, y, cv=5).mean()

    # Return accuracy #
    return accuracy


# Define the search space #
pbounds = {'rf_n_estimators': (50, 400),
           'rf_max_features': (0, 1),
           'rf_max_depth': (10, 40),
           'rf_min_samples_split': (2, 50),
           'rf_min_samples_leaf': (1, 50),
           'rf_bootstrap': (0, 1),
           'rf_max_leaf_nodes': (25, 100),
           'bag_n_estimators': (5, 100),
           'bag_max_samples': (10, 50),
           'bag_max_features': (1, 20)}


# Set the optimizer #
optimizer = BayesianOptimization(f=objective_bag, 
                                 pbounds=pbounds,
                                 random_state=1)


# Call the maximizer #
optimizer.maximize(init_points=50, n_iter=200)


# Pull best info #
best_hyperparameters = optimizer.max['params']
best_accuracy = optimizer.max['target']


# Comparison Matrix #
final = final.append({'Model': 'Bagged Random Forest',
                      'Accuracy': best_accuracy},
                      ignore_index=True)


###############################
#### XGBoost Random Forest ####
###############################

from xgboost import XGBClassifier


# Define objective function for Boosted Random Forest #
def objective_boost(n_estimators, max_depth, learning_rate, subsample,
              colsample_bytree, gamma, reg_alpha, reg_lambda):

    # Set the boosting classifier #
    boost = XGBClassifier(n_estimators = int(n_estimators),
                          max_depth = int(max_depth),
                          learning_rate = learning_rate,
                          subsample = subsample,
                          colsample_bytree = colsample_bytree,
                          gamma = gamma,
                          reg_alpha = reg_alpha, 
                          reg_lambda = reg_lambda)

    # Perform cross-validation and calculate mean accuracy
    accuracy = cross_val_score(boost, X_processed, y, cv=5).mean()

    # Return accuracy #
    return accuracy


# Define the search space #
pbounds = {
    'n_estimators': (50, 1000),
    'max_depth': (10, 100),
    'learning_rate': (0.000001, 1),
    'subsample': (0.1, 0.9),
    'colsample_bytree': (0.1, 0.9),
    'gamma': (0, 5),
    'reg_alpha': (0, 1),
    'reg_lambda': (0, 1)
}


# Set the optimizer #
optimizer = BayesianOptimization(f=objective_boost, pbounds=pbounds,
                                 random_state=1)


# Call the maximizer #
optimizer.maximize(init_points=50, n_iter=250)


# Pull best info #
best_hyperparameters = optimizer.max['params']
best_accuracy = optimizer.max['target']


# Comparison Matrix #
final = final.append({'Model': 'XGBoost Random Forest',
                      'Accuracy': best_accuracy},
                      ignore_index=True)


########################
#### Neural Network ####
########################

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, BatchNormalization


# Create a scikit learn wrapper for parameter tuning #
class MyKerasClassifier(KerasClassifier):
    def score(self, X_processed, y, sample_weight=None):
        _, accuracy = self.model.evaluate(X_processed, y, verbose=0)
        return accuracy


# Define objective function for Neural Network #
def objective_keras(batch_size, epochs, optimizer,
              learning_rate, num_hidden_layers1, num_hidden_layers2,
              num_nodes1, num_nodes2, activation):
    """
    Create objective function for the neural network hyperparamter tuning. 
    The code below alternates the optimizer and aciivation function. 
    The numeric hyperparameters are set below. 
    """
    
    # Set optimizer #
    if optimizer <= 0.25:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
    elif optimizer > 0.25 and optimizer <= 0.50:
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        
    elif optimizer > 0.50 and optimizer <= 0.75:
        optimizer = keras.optimizers.SGD(learning_rate = learning_rate)
        
    else:
        optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
        
    # Set activation function #
    if activation <= 0.25:
        activation = 'relu'
    elif activation > 0.25 and activation <= 0.5:
        activation = 'sigmoid'
    elif activation > 0.5 and activation <= 0.75:
        activation = 'tanh'
    else:
        activation = 'elu'
    
    # Instantiate model #
    model = Sequential()

    # Set input layer #
    model.add(Dense(int(num_nodes1), activation=activation,
                                 input_shape=(X_processed.shape[1],)))

    # Set hidden layer 1 with batch noramlizer #
    for _ in range(int(num_hidden_layers1)):
        model.add(Dense(int(num_nodes1), activation = activation))
        model.add(BatchNormalization())
        
    # Set hidden layer 2 with batch normalizer #
    for _ in range(int(num_hidden_layers2)):
        model.add(Dense(int(num_nodes2), activation = activation))
        model.add(BatchNormalization())
                   
    # Add the output layer #
    model.add(Dense(1, activation='sigmoid'))
    
    # Set the compile step #
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Create KerasClassifier wrapper
    model_wrapper = MyKerasClassifier(build_fn=lambda: model,
                                      batch_size=int(batch_size),
                                      epochs=int(epochs))


    # Perform cross-validation and calculate mean accuracy
    accuracy = cross_val_score(model_wrapper, X_processed, y, cv=5).mean()

    # Return accruacy #
    return accuracy


# Define the search space #
pbounds = {
    'batch_size': (100, 800),
    'epochs': (20, 100),
    'optimizer': (0, 1),
    'activation': (0, 1),
    'learning_rate': (0.0001, 0.3),
    'num_hidden_layers1': (1, 80),
    'num_hidden_layers2': (1, 40),
    'num_nodes1': (1, 50),
    'num_nodes2': (1, 25)
}


# Set the optimizer #
optimizer = BayesianOptimization(f=objective_keras, pbounds=pbounds,
                                 random_state=1)


# Call the maximizer #
optimizer.maximize(init_points=50, n_iter=400)


# Pull best info #
best_hyperparameters = optimizer.max['params']
best_accuracy = optimizer.max['target']


# Comparison Matrix #
final = final.append({'Model': 'Neural Network',
                      'Accuracy': best_accuracy},
                      ignore_index=True)


###############################
#### Best Model Prediction ####
###############################


# Import test data #
data_test = pd.read_csv('test.csv')


# Make copy #
test_with_id = data_test[['PassengerId']].copy()


# Drop Unneeded Variables #
X_test = data_test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1)


# Ordinal Encode #
X_test[['Sex', 'Embarked']] = ord_enc.transform(X_test[['Sex', 'Embarked']])


# KNN Imputer #
X_test_2 = pd.DataFrame(KNN_impute.transform(X_test), columns=X_test.columns)


# Replace Parch 9 as a 6 #
X_test_2['Parch'] = X_test_2['Parch'].replace(9, 6)


# Process data #
temp = processor.transform(X_test_2)


# Get categorical feature names #
enc_cat_features = list(processor.named_transformers_['cat']['encode']\
                        .get_feature_names_out())
    
    
# Concat label names #
labels = num_stand + enc_cat_features


# Make DF of processed data #
X_test_pro = pd.DataFrame(temp, columns=labels)

# Make integers for some columns #
best_hyperparameters['max_depth'] = int(best_hyperparameters['max_depth'])
best_hyperparameters['n_estimators'] = int(best_hyperparameters['n_estimators'])


# Predict #
best_boost = XGBClassifier(**best_hyperparameters)
best_boost.fit(X_processed, y)
predictions = best_boost.predict(X_test_pro)


# Put predctions into test dataset #
test_with_id['Survived'] = predictions


# Write dataset to CSV #
test_with_id.to_csv('titanic_submission.csv', index = False)

