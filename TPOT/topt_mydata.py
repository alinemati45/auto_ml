from tpot import TPOTClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('./data_image/All_data_precrocessing.csv')


features = tpot_data.drop('target', axis=1)
X_train, X_test, y_train, y_test  = train_test_split(features, tpot_data['target'], random_state=42)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=3, random_state=42  , n_jobs=-1,
                      use_dask=False)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
