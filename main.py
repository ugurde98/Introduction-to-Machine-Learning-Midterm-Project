import pandas as pd
from warnings import filterwarnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from time import time

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def matrixMetrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0][0]
    tp = cm[1][1]
    fn = cm[1][0]
    fp = cm[0][1]

    specificity = tn / (tn + fp)
    sensivity = tp / (tp + fn)

    matrixMetrics = pd.DataFrame({
        'Metrics': ['TP', 'TN', 'FP', 'FN', 'Specificity', 'Sensivity'],
        'Values': [tp, tn, fp, fn, specificity, sensivity]
    })

    return (matrixMetrics)


filterwarnings('ignore')

lb = pd.read_csv('labels.csv')
lb = lb.drop(labels=['Sample'], axis=1)

df = pd.read_csv('data.csv')
df = df.drop(df.columns[0], axis='columns')


# bağımlı ve bağımsız değişkenler belirlenmiştir.
y = lb[['disease_type']]
X = df.iloc[:, 1:]

# Veri seti %70 train, %30 test olarak bölünmüştür.
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# LightGBM Uygulaması
import lightgbm as lgb
def GradientBoostedTrees():
    t0 = time()
    lgb_model = lgb.LGBMClassifier()
    lgb_model.fit(X_train, y_train)
    # lgb_time = time() - t0
    lgb_acc = accuracy_score(y_test, lgb_model.predict(X_test))
    lgb_report = classification_report(y_test, lgb_model.predict(X_test))
    lgb_spec_sens = matrixMetrics(y_test, lgb_model.predict(X_test))
    lgb_auc_score = multiclass_roc_auc_score(y_test, lgb_model.predict(X_test))

    print("-----------------------LGB Algorithm Result Start------------------------")
    print("lgb_acc: ", lgb_acc)
    print("lgb_auc: ", lgb_auc_score)
    print(lgb_report)
    print(lgb_spec_sens)
    print("-----------------------LGB Algorithm Result End------------------------")

from sklearn.ensemble import RandomForestClassifier
def RandomForest():
    # Random Forest Uygulaması
    t0 = time()
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    # rf_time = time() - t0
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    rf_report = classification_report(y_test, rf_model.predict(X_test))
    rf_spec_sens = matrixMetrics(y_test, rf_model.predict(X_test))
    rf_auc_score = multiclass_roc_auc_score(y_test, rf_model.predict(X_test))

    print("-----------------------Random Forest Algorithm Result Start------------------------")
    print("rf_acc: ", rf_acc)
    print("rf_auc: ", rf_auc_score)
    print(rf_report)
    print(rf_spec_sens)
    print("-----------------------Random Forest Algorithm Result End------------------------")

print("aaaaa")
GradientBoostedTrees()
RandomForest()

print("xxxxx")







































