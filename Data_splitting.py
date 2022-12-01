#!/usr/bin/env python
import sys
import pandas as pd; import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
import scikitplot as skplt
from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import seaborn as sns; import matplotlib.pyplot as plt
import shap


'''
    1- Load and prepare data
'''
database = pd.read_csv('../Database/RW_DATA_CUT-OFF_17.06.2021.csv')
database = pd.DataFrame(database)

database.info

#Creating one dataframefor all the outcomes
Y_all = database[['DCR ', 'OSMesiCoded2Class','ORR ','OSMesiCoded24M','PFSMesi3' ,'TTFCoded3']] 


# 28 features at baseline
X_raw = database[['IT/CTIT', 'SurgeryY/N','AgeAtIO','Gender', 'SquamousNonSquamous', 'stadioAllaDiagnosi', 'fumoAllaDiagnosi','packYears', 'PDL1/1/149/50', 'LineaDiTerapiaICI', 'StageBasaleIO',  'BMIIoBasal', 'PS Baseline IO', 'TumorStage_IO', 'NodeStage_IO', 'MetastasisStage_IO', 'MLiverBasale', 'RTPreIT', 'LeukocytesIOBasal', 'NeutrophylsiAlBasale', 'MonocytesalBasale','LymphocytesAlBasale','NLR','LDHAlBasale', 'MBoneBasale','MPleuraBasale', 'MLinfonodiBasale', 'MSurreneBasale', 'MBrainBasale']]
X_raw.dtypes


print(X_raw)

#SPLITTING THA DATASET (stratify on DCR - clinically most important outcome)
X_train_raw, X_test_raw, Y_train_all, Y_test_all = train_test_split(X_raw, Y_all, stratify=Y_all['DCR '], random_state=84, test_size = 0.1)

#Creating separate files for all outcomes
Y_train_DCR=Y_train_all['DCR ']
Y_test_DCR=Y_test_all['DCR ']
Y_train_OS6=Y_train_all['OSMesiCoded2Class']
Y_test_OS6=Y_test_all['OSMesiCoded2Class']
Y_train_ORR=Y_train_all['ORR ']
Y_test_ORR=Y_test_all['ORR ']
Y_train_OS24=Y_train_all['OSMesiCoded24M']
Y_test_OS24=Y_test_all['OSMesiCoded24M']
Y_train_PFS3=Y_train_all['PFSMesi3']
Y_test_PFS3=Y_test_all['PFSMesi3']
Y_train_TTF3=Y_train_all['TTFCoded3']
Y_test_TTF3=Y_test_all['TTFCoded3']

with open('../Database/Final_datasets/datasplits.txt', 'w') as f1:
    print('Y_train_DCR:',Y_train_DCR.value_counts(), file=f1)
    print('Y_test_DCR:',Y_test_DCR.value_counts(), file=f1)
    print('Y_train_OS6:',Y_train_OS6.value_counts(), file=f1)
    print('Y_test_OS6:',Y_test_OS6.value_counts(), file=f1)
    print('Y_train_ORR:',Y_train_ORR.value_counts(), file=f1)
    print('Y_test_ORR:',Y_test_ORR.value_counts(), file=f1)
    print('Y_train_OS24:',Y_train_OS24.value_counts(), file=f1)
    print('Y_test_OS24:',Y_test_OS24.value_counts(), file=f1)
    print('Y_train_PFS3:',Y_train_PFS3.value_counts(), file=f1)
    print('Y_test_PFS3:',Y_test_PFS3.value_counts(), file=f1)
    print('Y_train_TTF3:',Y_train_TTF3.value_counts(), file=f1)
    print('Y_test_TTF3:',Y_test_TTF3.value_counts(), file=f1)

'''
    2- Impute missing data (using only the train set)
'''
# print total missing
print('Missing: %d' % X_train_raw.isnull().sum().sum())
# define imputer
imputer = IterativeImputer()
# fit on the dataset
imputer.fit(X_train_raw)
# transform the dataset
X_train = imputer.transform(X_train_raw)
X_train = pd.DataFrame(X_train, columns=X_train_raw.columns)
X_test = imputer.transform(X_test_raw)
X_test = pd.DataFrame(X_test, columns=X_test_raw.columns)

# print total missing
print('Missing: %d' % X_train.isnull().sum().sum())
print('Missing: %d' % X_test.isnull().sum().sum())


'''
    3- FEATURE SELECTION
'''
plt.figure(figsize=(30, 20))
heatmap = sns.heatmap(X_train.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.savefig('feature_correlation_heatmap.png', bbox_inches='tight')

cor_matrix = X_train.corr().abs()
print(cor_matrix)

upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.80)]
print(); print(to_drop)

#only one feature has correlation higher than .8 and is removed
X_train=X_train.drop(['MetastasisStage_IO'], axis = 1)
X_test=X_test.drop(['MetastasisStage_IO'], axis = 1)


'''
    4- Save the dataset after missing-value imputation
'''
# Input features (train and test)
X_train.to_csv('../Database/Final_datasets/X_train.csv')
X_test.to_csv('../Database/Final_datasets/X_test.csv')
# Outcomes (train and test)
Y_train_DCR.to_csv('../Database/Final_datasets/Y_train_DCR.csv')
Y_test_DCR.to_csv('../Database/Final_datasets/Y_test_DCR.csv')
Y_train_OS6.to_csv('../Database/Final_datasets/Y_train_OS6.csv')
Y_test_OS6.to_csv('../Database/Final_datasets/Y_test_OS6.csv')
Y_train_ORR.to_csv('../Database/Final_datasets/Y_train_ORR.csv')
Y_test_ORR.to_csv('../Database/Final_datasets/Y_test_ORR.csv')
Y_train_OS24.to_csv('../Database/Final_datasets/Y_train_OS24.csv')
Y_test_OS24.to_csv('../Database/Final_datasets/Y_test_OS24.csv')
Y_train_PFS3.to_csv('../Database/Final_datasets/Y_train_PFS3.csv')
Y_test_PFS3.to_csv('../Database/Final_datasets/Y_test_PFS3.csv')
Y_train_TTF3.to_csv('../Database/Final_datasets/Y_train_TTF3.csv')
Y_test_TTF3.to_csv('../Database/Final_datasets/Y_test_TTF3.csv')
