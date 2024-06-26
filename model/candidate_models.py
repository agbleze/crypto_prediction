#%%

from xgboost import XGBRFClassifier, XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, BaggingClassifier, 
                              HistGradientBoostingClassifier,ExtraTreesClassifier,
                              VotingClassifier, 
                              )
#from sklearn.neighbors import KNeighborsClassifier
from .preprocess_pipeline import PipelineBuilder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from typing import List, Tuple
from sklearn.model_selection import cross_val_score,cross_validate
import pandas as pd

#pipeline = PipelineBuilder()

#preprocess_pipeline = pipeline.build_data_preprocess_pipeline()

def get_candidate_classifiers(model_pipeline: PipelineBuilder, preprocess_pipeline):
    logit = LogisticRegression(class_weight='balanced')

    svc_rbf = SVC(kernel='rbf', class_weight='balanced', probability=True)
    svc_linear = SVC(kernel='linear', class_weight='balanced', probability=True)
    svc_poly = SVC(kernel='poly', class_weight='balanced', probability=True)
    rfc = RandomForestClassifier(class_weight='balanced')

    decision_tree = DecisionTreeClassifier(class_weight='balanced')
    extra_decision_tree = ExtraTreeClassifier(class_weight='balanced')
    logit_pipeline = model_pipeline.build_model_pipeline(model=logit, preprocess_pipeline=preprocess_pipeline)
    svc_rbf_pipeline = model_pipeline.build_model_pipeline(model=svc_rbf, preprocess_pipeline=preprocess_pipeline)
    svc_linear_pipeline = model_pipeline.build_model_pipeline(model=svc_linear, preprocess_pipeline=preprocess_pipeline)
    svc_poly_pipeline = model_pipeline.build_model_pipeline(model=svc_poly, preprocess_pipeline=preprocess_pipeline)
    rfc_pipeline = model_pipeline.build_model_pipeline(model=rfc, preprocess_pipeline=preprocess_pipeline)
    decision_tree_pipeline = model_pipeline.build_model_pipeline(model=decision_tree, preprocess_pipeline=preprocess_pipeline)
    extra_decision_tree_pipeline = model_pipeline.build_model_pipeline(model=extra_decision_tree, preprocess_pipeline=preprocess_pipeline)

    candidate_classifiers = {"Extra decision tree": extra_decision_tree_pipeline,
                            "Decision tree": decision_tree_pipeline,
                            "Radom forest classifier": rfc_pipeline,
                            "SVC poly": svc_poly_pipeline,
                            "SVC linear": svc_linear_pipeline,
                            "SVC rbf": svc_rbf_pipeline,
                            "Logistic regression": logit_pipeline
                            }
    return candidate_classifiers

# TODO:
#1. Add function to return best classifier -- done
#2. Add function to save best classifier  -- done
#3. Add function to bag best classifier
#4. Add function to boost best classifier
#5. Add function to plot residual plots for candidate models




# %%