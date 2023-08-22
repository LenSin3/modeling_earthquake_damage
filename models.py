# import machine learning processing modules
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# import binary classification machine learning modules
from sklearn.linear_model import LogisticRegression


# import modules for multi-class machine learning models
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
import xgboost as xgb
import catboost
import utils
SEED = 42

# instantiate binary classifiers
lr = LogisticRegression(random_state=SEED)
lin_svc = SVC(kernel='linear', random_state=SEED)

# instantiate multiclass classifiers
knn = KNeighborsClassifier()
rf  = RandomForestClassifier(n_estimators=1000, n_jobs = 4, random_state=SEED)
dt = DecisionTreeClassifier(random_state=SEED)
xgb_clf = xgb.XGBClassifier(objective='multi:softmax', n_estimators=1500, tree_method='gpu_hist', gpu_id=0, random_state = SEED)
ovr = OneVsRestClassifier(lr)
ovo = OneVsOneClassifier(lin_svc)

def classifiers_ensemble(type):
    if type == 'binary':
        # create tuple of classifier name and classifier
        classifiers = [
            ('LinearSVC', lin_svc),
            ('LogisticRegression', lr),
            ('KNearestNeighbors', knn),
            ('DecisionTreeClassifier', dt),
            ('RandomForestClassifier', rf)
        ]
    elif type == 'multi_class':
        # create tuple of classifier name and classifier
        classifiers = [
            # ('KNearestNeighbors', knn),
            # ('DecisionTreeClassifier', dt),
            ('RandomForestClassifier', rf)
            # ('OneVsRestLogisticRegression', ovr)
            # ('LinSVC OVO', ovo)
        ]
    else:
        raise utils.ClassifierType(type)
    return classifiers

def classifier_ensemble_hyperparameters(best_classifier):
    # list of classifier names
    classfiers_list = ['LinearSVC', 'LogisticRegression', 'KNearestNeighbors', 'DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier']
    # parameter dictionary
    params_dict = {
        'LinearSVC': [
            {'model': lin_svc},
            {'grid': {
                'linearsvc__C': [0.5, 1.0, 5.0, 10.0],
            }
            }
        ],
        'LogisticRegression': [
            {'model': lr},
            {'grid': {
                'logisticregression__penalty': ['l2'],
                'logisticregression__solver': ['newton-cg', 'sag', 'lbfgs', 'liblinear'],
                'logisticregression__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000]
            }
            }            
        ],
        'KNearestNeighbors': [
            {'model': knn},
            {'grid': {
                'knearestneighbors__n_neighbors': [3, 5, 8, 10],
                'knearestneighbors__radius': [0.5, 1.0, 5.0, 10.0],
                'knearestneighbors__algorithm': ['auto', 'ball_tree', 'kd_tree'],
                'knearestneighbors__leaf_size': [20, 30, 40, 60]
            }
            }
        ],
        'DecisionTreeClassifier': [
            {'model': dt},
            {'grid': {
                'decisiontreeclassifier__max_depth': [2, 4, 8, 10, 12, 16, 20],
                'decisiontreeclassifier__min_samples_leaf': [2, 4, 8, 10, 12, 16, 20]
            }
            }
        ],
        'RandomForestClassifier': [
            {'model': rf},
            {'grid' : {'randomforestclassifier__max_depth': [2, 15],
            'randomforestclassifier__min_samples_leaf': [1, 4],
            'randomforestclassifier__min_samples_split': [2, 6]
            # 'randomforestregressor__min_weight_fraction_leaf': [0.1, 0.3, 0.9],
            # 'randomforestregressor__max_features': ['auto', 'log2', 'sqrt', None]
            # 'randomforestregressor__max_leaf_nodes': [None, 10, 40, 90]
        }}],
        'XGBClassifier': [
            {'model': xgb_clf},
            {'grid' : {
            'xgbclassifier__learning_rate': [0.1, 0.01],
            'xgbclassifier__n_estimators': [2000, 3000, 5000],
            'xgbclassifier__max_depth': [3, 6],
            'xgbclassifier__min_child_weight': [6, 12],
            'xgbclassifier__gamma': [i/10.0 for i in range(3)]
        }}]
        
    }
    if best_classifier in classfiers_list:
        grid_params = params_dict[best_classifier][1]['grid']
        grid_model = params_dict[best_classifier][0]['model']
    else:
        print("{} is not among list of classifiers.".format(best_classifier))
    
    return grid_params, grid_model
