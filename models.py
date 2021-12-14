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
import xgboost
import catboost
import utils
SEED = 42

# instantiate binary classifiers
lr = LogisticRegression(random_state=SEED)
lin_svc = SVC(kernel='linear', random_state=SEED)

# instantiate multiclass classifiers
knn = KNeighborsClassifier()
rf  = RandomForestClassifier(random_state=SEED)
dt = DecisionTreeClassifier(random_state=SEED)
ovr = OneVsRestClassifier(lr)
ovo = OneVsOneClassifier(lin_svc)

def classifiers_ensemble(type):
    if type == 'binary':
        # create tuple of classifier name and classifier
        classifiers = [
            ('Linear SVC', lin_svc),
            ('Logistic Regression', lr),
            ('K Nearest Neighbors', knn),
            ('Classification Tree', dt),
            ('Forest Tree', rf)
        ]
    elif type == 'multi_class':
        # create tuple of classifier name and classifier
        classifiers = [
            ('K Nearest Neighbors', knn),
            ('Classification Tree', dt),
            ('Forest Tree', rf),
            ('LogReg OVR', ovr)
            # ('LinSVC OVO', ovo)
        ]
    else:
        raise utils.ClassifierType(type)
    return classifiers
