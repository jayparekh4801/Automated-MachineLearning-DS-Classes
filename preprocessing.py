import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PreProcessing :
    
    def __init__(self, *args) :
        self.phases = args  
    
    def fit(self, file) :
        self.dataset = pd.read_csv(file)    
        self.featureMetrics = self.dataset.iloc[ : , : -1].values
        self.resultMetrics = self.dataset.iloc[ : , -1].values
    
    def handleNulls(self) :
        self.imputer = SimpleImputer(missing_values="np.nan", strategy="mean")
        self.imputer.fit_transform(self.featureMetrics)
    
    def doSplit(self) :
        self.featureMetrics_train, self.featureMetrics_test, self.resultMetrics_train, self.resultMetrics_test = train_test_split(self.featureMetrics, self.resultMetrics, test_size=0.2, random_state=0)

    def doFeatureScaling(self) :
        scalar = StandardScaler()
        
