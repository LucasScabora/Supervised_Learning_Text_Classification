from copy import deepcopy

# Binary Classifiers: https://scikit-learn.org/stable/supervised_learning.html
# Multiclass Models: https://scikit-learn.org/stable/modules/multiclass.htm
from sklearn.linear_model     import (RidgeClassifier, LogisticRegression,
                                      SGDClassifier, PassiveAggressiveClassifier)
from sklearn.svm              import LinearSVC, SVC
from sklearn.neighbors        import (KNeighborsClassifier, NearestCentroid,
                                      RadiusNeighborsClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes      import GaussianNB, MultinomialNB, ComplementNB
from sklearn.tree             import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble         import (RandomForestClassifier, GradientBoostingClassifier,
                                      HistGradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.neural_network   import MLPClassifier
from lightgbm                 import LGBMClassifier


class AlgorithmModels:
    """
    Class for storing and retrieving scikit-learn models for binary and multiclass classification tasks.
    
    Attributes:
    - n_class (int): Number of classes in the dataset.
    - random_state (int): Seed for random number generation.
    - max_iter (int): Maximum number of iterations for some models.
    
    Methods:
    - get_all_binary_models() -> dict: Returns a dictionary of all binary classification models.
    - get_all_multiclass_models() -> dict: Returns a dictionary of all multiclass classification models.
    - get_binary_model(model_name: str): Returns the specified binary classification model.
    - get_multiclass_model(model_name: str): Returns the specified multiclass classification model.
    """
    # Algorithms and models employed for SKlearn
    def __init__(self, n_class, random_state:int = 91, max_iter:int = 80000):
        self.random_state = random_state
        self.max_iter = max_iter
        
        # Binary models
        self.binary_models = {
            'ComplementNB':                         ComplementNB(),
            'GaussianNB':                           GaussianNB(),
            'MultinomialNB':                        MultinomialNB(),
            'MLPClassifier':                        MLPClassifier(random_state=self.random_state,
                                                                  max_iter=self.max_iter),
            'RidgeClassifier':                      RidgeClassifier(random_state=self.random_state),
            'NearestCentroid':                      NearestCentroid(),
            'PassiveAggressiveClassifier':          PassiveAggressiveClassifier(max_iter=self.max_iter),
            'LinearSVC':                            LinearSVC(random_state=self.random_state,
                                                              max_iter=self.max_iter),
            'DecisionTreeClassifier':               DecisionTreeClassifier(random_state=self.random_state),
            'ExtraTreeClassifier':                  ExtraTreeClassifier(random_state=self.random_state),
            'ExtraTreesClassifier':                 ExtraTreesClassifier(random_state=self.random_state),
            'GaussianProcessClassifier':            GaussianProcessClassifier(random_state=self.random_state),
            'GradientBoostingClassifier':           GradientBoostingClassifier(random_state=self.random_state),
            'HistGradientBoostingClassifier':       HistGradientBoostingClassifier(random_state=self.random_state),
            'KNeighborsClassifier':                 KNeighborsClassifier(),
            'LGBMClassifier':                       LGBMClassifier(objective='binary', num_classes=n_class-1, # positive
                                                                   random_state=self.random_state),
            'LogisticRegression':                   LogisticRegression(random_state=self.random_state),
            'RadiusNeighborsClassifier':            RadiusNeighborsClassifier(outlier_label='most_frequent'),
            'RandomForestClassifier':               RandomForestClassifier(random_state=self.random_state),
            'SGDClassifier':                        SGDClassifier(max_iter=self.max_iter,
                                                                  random_state=self.random_state),
            'LinearSVC':                            SVC(kernel='linear',
                                                        random_state=self.random_state,
                                                        decision_function_shape='ovo',
                                                        max_iter=self.max_iter,
                                                        probability=True),
        }
        
        # Multiclass models
        self.multiclass_models = {
            'MLPClassifier':                   MLPClassifier(random_state=self.random_state,
                                                             max_iter=self.max_iter),
            'DecisionTreeClassifier':          DecisionTreeClassifier(random_state=self.random_state),
            'ExtraTreeClassifier':             ExtraTreeClassifier(random_state=self.random_state),
            'ExtraTreesClassifier':            ExtraTreesClassifier(random_state=self.random_state),
            'RandomForestClassifier':          RandomForestClassifier(random_state=self.random_state),
            'KNeighborsClassifier':            KNeighborsClassifier(),
            'RadiusNeighborsClassifier':       RadiusNeighborsClassifier(outlier_label='most_frequent'),
            'LGBMClassifier':                  LGBMClassifier(objective='multiclass',
                                                              num_classes=n_class,
                                                              verbose= -100,
                                                              random_state=self.random_state),
            'LinearSVC':                       SVC(kernel='linear', random_state=self.random_state,
                                                   decision_function_shape='ovo',
                                                   max_iter=self.max_iter,
                                                   probability=True)
        }
        
    def get_all_binary_models(self) -> dict:
        return deepcopy(self.binary_models)

    def get_all_multiclass_models(self) -> dict:
        return deepcopy(self.multiclass_models)

    def get_binary_model(self, model_name:str):
        return deepcopy(self.binary_models[model_name])

    def get_multiclass_model(self, model_name:str):
        return deepcopy(self.multiclass_models[model_name])
