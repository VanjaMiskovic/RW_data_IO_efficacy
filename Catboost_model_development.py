#!/usr/bin/env python
import sys
import pandas as pd; import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import scikitplot as skplt
from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import seaborn as sns; import matplotlib.pyplot as plt
import shap
# matplotlib to plot the curve
plt.style.use('seaborn')

'''
    1- Load and prepare the data
'''
#Import datasets
X_train = pd.read_csv('dataset/X_train.csv') 
X_test = pd.read_csv('dataset/X_test.csv')
Y_train_DCR = pd.read_csv('dataset//Y_train_DCR.csv')
Y_test_DCR = pd.read_csv('dataset/Y_test_DCR.csv')

# Extract the features
X_train=X_train[['IT/CTIT', 'PDL1/1/149/50', 'LineaDiTerapiaICI', 'StageBasaleIO',  'BMIIoBasal', 'PS Baseline IO', 'NodeStage_IO', 'MLiverBasale',  'LeukocytesIOBasal', 'NeutrophylsiAlBasale', 'MonocytesalBasale', 'LymphocytesAlBasale', 'NLR','LDHAlBasale', 'SquamousNonSquamous', 'SurgeryY/N', 'AgeAtIO', 'Gender', 'stadioAllaDiagnosi', 'fumoAllaDiagnosi', 'TumorStage_IO', 'RTPreIT', 'MBoneBasale', 'MPleuraBasale', 'MLinfonodiBasale', 'MSurreneBasale', 'MBrainBasale']]
X_test=X_test[['IT/CTIT', 'PDL1/1/149/50', 'LineaDiTerapiaICI', 'StageBasaleIO',  'BMIIoBasal', 'PS Baseline IO', 'NodeStage_IO', 'MLiverBasale', 'LeukocytesIOBasal', 'NeutrophylsiAlBasale', 'MonocytesalBasale', 'LymphocytesAlBasale', 'NLR', 'LDHAlBasale', 'SquamousNonSquamous', 'SurgeryY/N', 'AgeAtIO', 'Gender', 'stadioAllaDiagnosi', 'fumoAllaDiagnosi', 'TumorStage_IO', 'RTPreIT', 'MBoneBasale', 'MPleuraBasale', 'MLinfonodiBasale', 'MSurreneBasale', 'MBrainBasale']]

Y_train_DCR = Y_train_DCR['DCR '] #change the outcome
Y_test_DCR = Y_test_DCR['DCR ']   #change the outcome

#converting categorical features into int
X_train[['IT/CTIT',  'SquamousNonSquamous', 'PDL1/1/149/50', 'LineaDiTerapiaICI', 'StageBasaleIO', 'PS Baseline IO', 'NodeStage_IO',  'MLiverBasale','SurgeryY/N','Gender','stadioAllaDiagnosi', 'fumoAllaDiagnosi', 'MLinfonodiBasale', 'MSurreneBasale', 'MBrainBasale','TumorStage_IO', 'RTPreIT', 'MBoneBasale','MPleuraBasale']]=X_train[['IT/CTIT',  'SquamousNonSquamous', 'PDL1/1/149/50', 'LineaDiTerapiaICI', 'StageBasaleIO', 'PS Baseline IO', 'NodeStage_IO',  'MLiverBasale','SurgeryY/N','Gender','stadioAllaDiagnosi', 'fumoAllaDiagnosi', 'MLinfonodiBasale', 'MSurreneBasale', 'MBrainBasale','TumorStage_IO', 'RTPreIT','MBoneBasale','MPleuraBasale']].astype(int)

print(X_train.dtypes)

#selecting categorical features for CatBoost
categorical_features_indices = np.where(X_train.dtypes == 'int32')[0]
X_train.dtypes
categorical_features_indices=categorical_features_indices.astype(int) 
categorical_features_indices

#scaling continuous features between 0 and 1
sc = MinMaxScaler()
X_train[['AgeAtIO','BMIIoBasal','LeukocytesIOBasal','NeutrophylsiAlBasale','MonocytesalBasale','LymphocytesAlBasale','NLR','LDHAlBasale']] = sc.fit_transform(X_train[['AgeAtIO','BMIIoBasal','LeukocytesIOBasal','NeutrophylsiAlBasale','MonocytesalBasale','LymphocytesAlBasale','NLR','LDHAlBasale']])

'''
    2- Model definition with CatBoost
'''
clf = CatBoostClassifier()

# defining the hyper-parameters grid for Cross Validation (CV)
grid = {
    'learning_rate': [0.001,0.003, 0.03],
    'depth': [4,6,7],
    'l2_leaf_reg': [7,9,11,13]
    }
# Use F1 score for model evaluation during training
scorer = make_scorer(f1_score) 
   
#performing gridsearch
clf_grid = GridSearchCV(estimator=clf,
                        param_grid=grid,
                        scoring=scorer,             # F1 score
                        refit=True,                 # re-do training with the best parameters
                        cv=10,                      # 10 fold
                        return_train_score=True
)

# Fit the model via grid search (10-fold CV)
# NOTE: this may take some time...
grid_search_result = clf_grid.fit(X_train, Y_train_DCR)

best_param = clf_grid.best_params_
print(best_param)
print(pd.DataFrame(grid_search_result.cv_results_))  

results=pd.DataFrame(clf_grid.cv_results_)

#defining the model with best hyperparameters 
model = CatBoostClassifier(iterations=1000,
                           learning_rate=best_param['learning_rate'],
                           loss_function='Logloss',
                           depth=best_param['depth'],
                           l2_leaf_reg=best_param['l2_leaf_reg'],
                           eval_metric='F1',
                           leaf_estimation_iterations=20,
                           use_best_model=False,
                           logging_level='Silent',
                           random_seed=42,
                           cat_features=categorical_features_indices
)

train_pool = Pool(X_train, Y_train_DCR, cat_features=categorical_features_indices)
#model fitting
model.fit(train_pool)

'''
    3- Model evaluation
'''
#showing metrics on training set
Y_pred_training_DCR = model.predict(X_train)
from sklearn import metrics
print('The accuracy on training is :\t', metrics.accuracy_score(Y_train_DCR, Y_pred_training_DCR))

# cross_validation 
scorer1 = make_scorer(accuracy_score)
scores = cross_val_score(model, X_train, Y_train_DCR, scoring=scorer1, cv=10)
scorer2 = make_scorer(f1_score)
scores2 = cross_val_score(model, X_train, Y_train_DCR, scoring=scorer2, cv=10)

#display scores
def display_scores(scores):
    print('Acc. Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())
display_scores(scores)

def display_scores_f1(scores2):
    print('F1: Scores:', scores2)
    print('Mean:', scores2.mean())
    print('Standard deviation:', scores2.std())
display_scores_f1(scores2)

#scaling continous features in the test set between 0 and 1
X_test[['AgeAtIO','BMIIoBasal','LeukocytesIOBasal','NeutrophylsiAlBasale','MonocytesalBasale','LymphocytesAlBasale','NLR','LDHAlBasale']] = sc.transform(X_test[['AgeAtIO','BMIIoBasal','LeukocytesIOBasal','NeutrophylsiAlBasale','MonocytesalBasale','LymphocytesAlBasale','NLR','LDHAlBasale']])

#predicting test set
Y_pred_DCR = model.predict(X_test)

#showing metrics on test
print('The accuracy on testing is :\t', metrics.accuracy_score(Y_test_DCR, Y_pred_DCR))
print(classification_report(Y_test_DCR, Y_pred_DCR))

#Feature importance
feature_importances = model.get_feature_importance(train_pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))


# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(Y_test_DCR, Y_pred_DCR, pos_label=1)
random_probs = [0 for i in range(len(Y_test_DCR))]
p_fpr, p_tpr, _ = roc_curve(Y_test_DCR, random_probs, pos_label=1)
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(Y_test_DCR, Y_pred_DCR)
print('The AUC is :\t', auc_score)

# plot roc curves
plt.figure(figsize=(10, 5))
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='CatBoost')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('results/ROC_DCR_B',dpi=300)
plt.show()

#plotting ConfusionMatrix 
cf_matrix = confusion_matrix(Y_test_DCR, Y_pred_DCR)
df_cm = pd.DataFrame(cf_matrix, columns=np.unique(Y_test_DCR), index = np.unique(Y_pred_DCR))
df_cm.index.name = 'Actual response'
df_cm.columns.name = 'Predicted response'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm,
                 cmap="YlGnBu",
                 annot=True,
                 annot_kws={"size": 16},
                 vmin=0,
                 vmax=35
                )
plt.savefig('results/confusionmatrix_DCR.png',dpi=300)

#renaming columns for SHAP plot
X_train.rename(columns ={'IT/CTIT':'IO/IOCT', 'SurgeryY/N':'Surgery', 'AgeAtIO':'Age', 'Gender':'Sex', 'SquamousNonSquamous':'Histology', 'stadioAllaDiagnosi':'TNMd', 'fumoAllaDiagnosi':'smoke', 'PDL1/1/149/50':'PDL1', 'LineaDiTerapiaICI':'Nr. Line of IO', 'StageBasaleIO':'TNMio', 'BMIIoBasal':'BMI', 'PS Baseline IO':'ECOG PS', 'TumorStage_IO':'T', 'NodeStage_IO':'N', 'MLiverBasale':'Liver mets', 'RTPreIT':'RT', 'LeukocytesIOBasal':'ALC', 'NeutrophylsiAlBasale':'ANC', 'MonocytesalBasale':'AMC', 'LymphocytesAlBasale':'ALyC', 'NLR':'NLR', 'LDHAlBasale':'LDH', 'MBoneBasale':'Bone mets', 'MPleuraBasale':'Pleura mets', 'MLinfonodiBasale':'Lymph nodes mets', 'MSurreneBasale':'Adrenal mets', 'MBrainBasale':'Brain mets'}, inplace = True)
X_test.rename(columns ={'IT/CTIT':'IO/IOCT', 'SurgeryY/N':'Surgery', 'AgeAtIO':'Age', 'Gender':'Sex', 'SquamousNonSquamous':'Histology', 'stadioAllaDiagnosi':'TNMd', 'fumoAllaDiagnosi':'smoke', 'PDL1/1/149/50':'PDL1', 'LineaDiTerapiaICI':'Nr. Line of IO', 'StageBasaleIO':'TNMio', 'BMIIoBasal':'BMI', 'PS Baseline IO':'ECOG PS', 'TumorStage_IO':'T', 'NodeStage_IO':'N', 'MLiverBasale':'Liver mets', 'RTPreIT':'RT', 'LeukocytesIOBasal':'ALC',  'NeutrophylsiAlBasale':'ANC', 'MonocytesalBasale':'AMC', 'LymphocytesAlBasale':'ALyC', 'NLR':'NLR', 'LDHAlBasale':'LDH', 'MBoneBasale':'Bone mets', 'MPleuraBasale':'Pleura mets', 'MLinfonodiBasale':'Lymph nodes mets', 'MSurreneBasale':'Adrenal mets', 'MBrainBasale':'Brain mets'}, inplace = True)

'''
    4- Explainability
'''
#SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
len(shap_values)

#SHAP summary plot
plt.figure(figsize=(30, 10))
shap.plots.beeswarm(shap_values, max_display=28)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
plt.savefig('results/DCR_SHAP_B.png', bbox_inches='tight')


def global_shap_importance(model, X):
    """ Return a dataframe containing the features sorted by Shap importance
    Parameters
    ----------
    model : The tree-based model
    X : pd.Dataframe
         training set/test set/the whole dataset ... (without the label)
    Returns
    -------
    pd.Dataframe
        A dataframe containing the features sorted by Shap importance
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    cohorts = {"": shap_values}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
    feature_importance = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=['features', 'importance'])
    feature_importance.sort_values(
        by=['importance'], ascending=False, inplace=True)
    return feature_importance

global_shap_importance(model, X_test)

with open('results/results_DCR.txt', 'w') as f2:
    print('The accuracy on training is :\t', metrics.accuracy_score(Y_train_DCR, Y_pred_training_DCR),file=f2)
    print('The scores on 10 cross_val are:\t', display_scores(scores),file=f2)
    print('The accuracy on testing is :\t', metrics.accuracy_score(Y_test_DCR, Y_pred_DCR), file=f2)
    print(classification_report(Y_test_DCR, Y_pred_DCR), file=f2)
    print('The accuracy on testing is :\t', metrics.accuracy_score(Y_test_DCR, Y_pred_DCR), file=f2)
    print('The AUC is :\t', auc_score, file=f2)


