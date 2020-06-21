# SupervisedMachineLearning

## Project Overview
The following repository includes Jupyter notebooks for supervised machine learning models to assess credit risk, using data from LendingClub, a peer-to-peer lending services company.

Credit risk inherently is unbalanced in terms of classification (as the number of good loans outnumbers risky loans by a large volume). Therefore, machine learning models were built using different resampling techniques from imbalanced-learn and scikit-learn libiraries.

The analysis of the models' performance is included below.

As an extension to the resampling analyis, two additional models with ensemble classifiers were trained. Results are discussed below.

## Resources
Data Sources: LoanStats_2019Q1.csv
Language: Python
Libraries: Warnings, NumPy, Pandas, Scikit-learn, Imbalanced-learn, Operator, Pathlib, Collections
Software: Jupyter Lab 1.2.6

## Files
The Challenge folder contains the notebooks related to the credit risk analysis described in the overview, above.

The different resampling models were trained in the Jupyter notebook "credit_risk_resampling.ipynb". For each resampling technique, different solvers were used so that the model with the best overall results for detecting high_risk could be identified.

The different models with ensemble classifiers were trained in the Jupyter notebook "credit_risk_ensemble.ipynb".

The Practice folder contains practice files.

## Results

### Resampling results: Oversampling using RandomOverSampler
Results for each solver are summarized in the tables below.

*solver = newton-cg*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.02 | 0.77 | 0.04 |
|low_risk| 1.00 | 0.76 | 0.86 |

*solver = lbfgs*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.73 | 0.04 |
|low_risk| 1.00 | 0.57 | 0.72 |

*solver = liblinear*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.02 | 0.78 | 0.04 |
|low_risk| 1.00 | 0.75 | 0.86 |

*solver = sag*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.51 | 0.02 |
|low_risk| 1.00 | 0.63 | 0.77 |

*solver = saga*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.52 | 0.02 |
|low_risk| 1.00 | 0.62 | 0.76 |

RandomOverSampler with any of the solvers is good at predicting low_risk and poor at predicting high_risk, as evidenced by the precision values.

Recall values are more varied, with newton-cg and liblinear solvers producing the best results for both high_risk and low_risk. The lbfgs solver produces relatively strong recall values for high_risk, but weaker for low_risk. Both sag and saga solvers produce poor recall values for both high_risk and low_risk.

The following table shows the RandomOverSampler's balanced accuracy score for each solver:

| Solver | Balanced Accuracy Score |
|--------|-------------------------|
| newton-cg | 0.76 |
| lbfgs | 0.65 |
| liblinear | **0.77** |
| sag | 0.57 |
| saga | 0.57 |

The RandomOverSampler algorithm with solver equal to *liblinear* has the highest balanced accuracy score by comparison with the other solvers.

### Resampling results: Oversampling using SMOTE
Results for each solver are summarized in the tables below.

*solver = newton-cg*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.04 | 0.54 | 0.08 |
|low_risk| 1.00 | 0.93 | 0.96 |

*solver = lbfgs*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.63 | 0.02 |
|low_risk| 1.00 | 0.68 | 0.81 |

*solver = liblinear*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.68 | 0.03 |
|low_risk| 1.00 | 0.69 | 0.82 |

*solver = sag*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.47 | 0.02 |
|low_risk| 1.00 | 0.67 | 0.80 |

*solver = saga*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.46 | 0.02 |
|low_risk| 1.00 | 0.66 | 0.79 |

SMOTE with any of the solvers is good at predicting low_risk and poor at predicting high_risk, as evidenced by the precision values.

Recall values are not very impressive with the exception of the newton-cg solver for low risk.

The following table shows the SMOTE's balanced accuracy score for each solver:

| Solver | Balanced Accuracy Score |
|--------|-------------------------|
| newton-cg | **0.74** |
| lbfgs | 0.66 |
| liblinear | 0.69 |
| sag | 0.57 |
| saga | 0.56 |

The SMOTE algorithm with solver equal to *newton-cg* has the highest balanced accuracy score by comparison with the other solvers.

### Resampling results: Undersampling using ClusterCentroids
Results for each solver are summarized in the tables below.

*solver = newton-cg*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.93 | 0.02 |
|low_risk| 1.00 | 0.31 | 0.48 |

*solver = lbfgs*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.68 | 0.01 |
|low_risk| 1.00 | 0.41 | 0.58 |

*solver = liblinear*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.85 | 0.02 |
|low_risk| 1.00 | 0.47 | 0.64 |

*solver = sag*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.13 | 0.02 |
|low_risk| 0.99 | 0.94 | 0.97 |

*solver = saga*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.02 | 0.02 |
|low_risk| 0.99 | 0.99 | 0.99 |

ClusterCentroids with any of the solvers is good at predicting low_risk and poor at predicting high_risk, as evidenced by the precision values.

Recall values are more varied. The newton-cg solver produces the best results for high_risk; however, the results for low_risk are poor. The liblinear solver performs similarly to the newton-cg solver, with high_risk recall still quite strong, and low_risk stronger than newton_cg. The lbfgs solver produces recall values that are not particularly impressive, with better results for high_risk. Both sag and saga solvers produce poor recall values for high_risk and very strong results for low_risk.

The following table shows the ClusterCentroids's balanced accuracy score for each solver:

| Solver | Balanced Accuracy Score |
|--------|-------------------------|
|newton-cg | 0.62 |
| lbfgs | 0.55 |
| liblinear | **0.66** |
| sag | 0.53 |
| saga | 0.51 |

The ClusterCentroids algorithm with solver equal to *liblinear* has the highest balanced accuracy score by comparison with the other solvers.

### Resampling results: Combination Sampling using SMOTEENN
Results for each solver are summarized in the tables below.

*solver = newton-cg*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.04 | 0.58 | 0.07 |
|low_risk| 1.00 | 0.91 | 0.95 |

*solver = lbfgs*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.78 | 0.02 |
|low_risk| 1.00 | 0.54 | 0.70 |

*solver = liblinear*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.76 | 0.03 |
|low_risk| 1.00 | 0.69 | 0.81 |

*solver = sag*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.71 | 0.02 |
|low_risk| 1.00 | 0.51 | 0.67 |

*solver = saga*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.01 | 0.70 | 0.02 |
|low_risk| 1.00 | 0.49 | 0.66 |

SMOTEENN with any of the solvers is good at predicting low_risk and poor at predicting high_risk, as evidenced by the precision values.

Recall values are more varied. The newton-cg solver produces recall results very well for low_risk, and somewhat poorly for high risk. The other solvers generally perform similarly to one another, with better results for high_risk and somewhat mediocre results for low_risk.

The following table shows the SMOTEENN's balanced accuracy score for each solver:

| Solver | Balanced Accuracy Score |
|--------|-------------------------|
| newton-cg | **0.75** |
| lbfgs | 0.66 |
| liblinear | 0.73 |
| sag | 0.61 |
| saga | 0.60 |

The SMOTEENN algorithm with solver equal to *newton-cg* has the highest balanced accuracy score by comparison with the other solvers.

#### Resampling Conclusions
Of all the models trained, the best model for generating predictions related to credit risk is RandomOverSampler with solver equal to liblinear. This resampling model represents the best combination of high recall for high_risk and low_risk.

The rationale for this choice is as follows:
- When dealing with credit risk, the most important thing is to identify high_risk
- Therefore, it makes the most sense to select a model with a high recall for high_risk
- However, the model should still have a high recall for low_risk, if possible, since in reality, a false negative result for low_risk represents a loan that could be sold to the client
- Any false negative for low_risk or true positive for high_risk could, in theory be submitted to a manual review process, where an underwriter could evaluate the credit risk
- The key takeaway is to avoid false negatives for high_risk, since the bank would be taking a risk issuing these loans
- Since all models perform similarly with respect to precision, precision is not a factor that has influenced this conclusion

### Ensemble learning results
Results for each ensemble classifier are summarized in the tables below.

*BalancedRandomForestClassifier*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.03 | 0.77 | 0.05 |
|low_risk| 1.00 | 0.88 | 0.94 |

*EasyEnsembleClassifier*
|         | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
|high_risk| 0.06 | 0.94 | 0.12 |
|low_risk| 1.00 | 0.94 | 0.97 |

The following table shows the balanced accuracy score for each ensemble classifier:

| Ensemble Classifier | Balanced Accuracy Score |
|---------------------|-------------------------|
| BalancedRandomForestClassifier | 0.82 |
| EasyEnsembleCLassifier | **0.94** |

The EasyEnsembleCLassifier algorithm has the higher balanced accuracy score by comparison with that of the BalancedRandomForestClassifier.

#### Ensemble Conclusions
The EasyEnsembleClassifier outperforms the BlancedRandomForestClassifier on all metrics. Both models perform better by cimparison with the Resampling models discussed above.

Of all the models trained, the best model for generating predictions related to credit risk is EasyEnsembleClassifier. This model represents the best combination of high recall for high_risk and low_risk. While precision remains weak for high_risk, it is still better than all other models trained.

## Recommendations
Overall, the best resampling model was RandomOverSampler with liblinear solver, and the best ensemble model was EasyEnsembleClassifier. The best model overall was the EasyEnsembleClassifier model.

On the assumption that the underwriters have the capacity to review the loan applications which were predicted as high_risk, should the be referred for manual review, the recommendation would be to go with the EasyEnsembleClassifier model.

However, if the underwriting team would not have the capacity to review all the referred loans based on the current model, it is recommended that another model be trained so that the work volume of the underwriting team could be more manageable. The new model would strive to maintain strong recall values, and increase precision for high_risk, since the false positives for high_risk contribute to the number of loan applications forwarded to the underwriting team for manual review.
