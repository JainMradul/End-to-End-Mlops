# MLOPS - Heart Failure Prediction

Capstone project for the Machine Learning Engineer with Microsoft Azure Nanodegree

Data Set used : [heart failure prediction dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) available from Kaggle.

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

## Dataset

### Overview
The dataset, downloaded from Kaggle, contains medical data of patients of various ages and different genders. 

Features : 12

Independent Variables : Age, Gender, anaemia (categorical: 0 or 1), creatinine phosphate (numeric: Level of the CPK enzyme in the blood (mcg/L)) and ejection fraction (numeric: Percentage of blood leaving the heart at each contraction). 

These variables are used to predict whether the patient will survive the heart failure or not (Death Event: 0 - no or 1 - yes).


### Task
Classification task to predict death event by heart failure, expressed by the DEATH_EVENT variable with a binary (0 or 1) outcome

### Access
The dataset is uploaded and registered in tabular form on the ML Azure workspaceblobstore. The dataset URL/datastore path can be used to access the data in the Jupyter notebooks used for model training via python SDK.

![dataset](https://github.com/JainMradul/End-to-End-Mlops/blob/main/screenshots/dataset.PNG)

## Automated ML
The AutoML notebook uses the Python SDK to train a variety of models and arrive at the best metric - accuracy in our case. 

A standard CPU-based compute cluster with 6 nodes is deployed to run the experiments.

### Results
The AutoML run trained a number of models and gave a best accuracy of around 87% by the Voting Ensemble classifier (the best model). 

The parameters are experiment timeout of 30 minutes with max concurrent iterations at 6. 

Dataset passes all the GuardRail and AutoML uses various gradient boosting ensembles to arrive at the best metric, there don't seem to be any needs for improvement as such but different evaluation metrics along with more experiment time could be used to identify the best model - in our case we have used "accuracy" and 30 minutes

![automl](https://github.com/JainMradul/End-to-End-Mlops/blob/main/screenshots/automl1.PNG)
![automl](https://github.com/JainMradul/End-to-End-Mlops/blob/main/screenshots/automl2.PNG)
![automl](https://github.com/JainMradul/End-to-End-Mlops/blob/main/screenshots/automl3.PNG)

## Hyperparameter Tuning
Hyperdrive is used with SKLearn's Logistic Regression to accomplish classification task.

Following two hyperparameters has been tuned:

* C - inverse of regularisation strength, ie higher C indicates lower regularisation strength. C range [0.1-1]
* Max iterations - Maximum number of training iterations per child run, range [25,50,75,100,125]

The parameters were sampled using Random Sampling, with Bandit Policy for an early termination.

### Results
The best model outputs an accuracy of around 91.1%, with a regularisation strength of 0.5. Since this accuracy is higher than that of the AutoMl model hence this model has been deployed

![hyperdrive](https://github.com/JainMradul/End-to-End-Mlops/blob/main/screenshots/hd1.PNG)
![hyperdrive](https://github.com/JainMradul/End-to-End-Mlops/blob/main/screenshots/hd2.PNG)
![hyperdrive](https://github.com/JainMradul/End-to-End-Mlops/blob/main/screenshots/hd3.PNG)
![hyperdrive](https://github.com/JainMradul/End-to-End-Mlops/blob/main/screenshots/hd4.PNG)
![hyperdrive](https://github.com/JainMradul/End-to-End-Mlops/blob/main/screenshots/hd5.PNG)


## Model Deployment
Hyperdrive model is deployed as it had a higher accuracy. The model was registered and deployed as an Azure Container Instance (ACI) webservice, with insights enabled.
Sample test data has been converted to JSON format and passed to the deployed ACI service to test the endpoint

![dep1](https://github.com/JainMradul/End-to-End-Mlops/blob/main/screenshots/dep1.PNG)
![dep2](https://github.com/JainMradul/End-to-End-Mlops/blob/main/screenshots/dep2.PNG)
![dep3](https://github.com/JainMradul/End-to-End-Mlops/blob/main/screenshots/dep3.PNG)

## Screen Recording
https://youtu.be/Ng1ft5gCkFM
