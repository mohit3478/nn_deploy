import numpy as np 
import os
from src.config import config 

class preprocess_data:
    
    def fit(self,X,y=None):
        
        self.num_rows =X.shape[0]
        self.num_input_features = X.shape[1]
        
        if len(X.shape) == 1:
          self.target_feature_dim = 1
        else:
            self.target_feature_dim = X.shape[1]
        
        if len(y.shape) == 1:
          self.target_feature_dim = 1
        else:
            self.target_feature_dim = y.shape[1]
        
        
    def transfrom(self ,X=None,y=None):
        
         
        self.X = np.array(X).reshape(self.num_rows,self.num_input_features)
        self.Y = np.array(y).reshape(self.num_rows,self.target_feature_dim)
        
        return self.X, self.Y