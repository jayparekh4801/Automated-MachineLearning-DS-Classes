import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class PreProcessing :
    
    def __init__(self, *args) :
        self.phases = args  
    
    def fit(self, file) :
        self.dataset = pd.read_csv(file)    
        self.featureMetrics = self.dataset.iloc[ : , : -1].values
        self.resultMetrics = self.dataset.iloc[ : , -1].values
    
    def handleNulls(self, missing = "np.nan", strategy = "mean") :
        self.imputer = SimpleImputer(missing_values=missing, strategy=strategy)
        self.imputer.fit_transform(self.featureMetrics)
    
    def doSplit(self, test = 0.2, state = 0) :
        self.featureMetrics_train, self.featureMetrics_test, self.resultMetrics_train, self.resultMetrics_test = train_test_split(self.featureMetrics, self.resultMetrics, test_size=test, random_state=state)

    def doFeatureScaling(self, method = "StandardScaler") :
        if(method == "StandardScaler") :
            scaler = StandardScaler()
            self.featureMetrics_train = scaler.fit_transform(self.featureMetrics_train)
            self.featureMetrics_test = scaler.transform(self.featureMetrics_test)
        
        else :
            scaler = Normalizer()
            self.featureMetrics_train = scaler.fit_transform(self.featureMetrics_train)
            self.featureMetrics_test = scaler.transform(self.featureMetrics_test)

    def handleCategoricalData(self, encoding = "onehotencoding", label = 0, remain = "passthrough") :
        if(encoding == "onehotencoding" ) :
            ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [label])], remainder=remain)
            self.featureMetrics_train = ct.fit_transform(self.featureMetrics_train)
            
        
