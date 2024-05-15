#%% import all models required for the analysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
#from argparse import Namespace
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns


one = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

class PipelineBuilder(object):
    def __init__(self, num_features: list,
                 categorical_features: list
                 ):
        self.num_features = num_features
        self.categorical_features = categorical_features
   
    
    @classmethod
    def build_data_preprocess_pipeline(self):
        self.preprocess_pipeline =  make_column_transformer((scaler, self.num_features),
                                                        #(one, self.categorical_features)
                                                      )
        
        return self.preprocess_pipeline
        
    
    @classmethod
    def build_model_pipeline(self, model = None, 
                             preprocess_pipeline = None,
                             class_weight='balanced'
                             ):
        if (model == None):
            model = LogisticRegression(class_weight=class_weight)
            
        if (preprocess_pipeline == None):
            if not hasattr(PipelineBuilder, "preprocess_pipeline"):
                preprocess_pipeline = self.build_data_preprocess_pipeline()
            else:
                preprocess_pipeline = self.preprocess_pipeline
            
        model_pipeline = make_pipeline(preprocess_pipeline,
                                             model
                                            )         
            
        return model_pipeline
        