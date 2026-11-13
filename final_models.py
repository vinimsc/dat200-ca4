# Group 3
# Stig Espen Paulsberg
# Vinicius Millan
# Zaur Kambiev

###########################################
#                Imports                  #
###########################################

from pathlib import Path

import pandas as pd

from sklearn.metrics.pairwise   import laplacian_kernel
from sklearn.pipeline           import Pipeline
from sklearn.preprocessing      import StandardScaler
from sklearn.svm                import SVC
from xgboost                    import XGBClassifier
from sklearn.compose            import ColumnTransformer
from sklearn.decomposition      import PCA

###########################################
#               Read files                #
###########################################

current_folder = Path(__file__).parent
data_folder = current_folder / "data"

train_df = pd.read_csv(data_folder / "trainCA4DAT200.csv", index_col="CustomerId")
test_df = pd.read_csv(data_folder / "testCA4DAT200.csv", index_col="CustomerId")

###########################################
#               Preprocessing             #
###########################################

def encode_binary_features(df):
    new_df = df.copy()
    new_df["InternationalPlan"] = (df.InternationalPlan == 'yes').astype(float)
    new_df["VoiceMailPlan"] = (df.VoiceMailPlan == 'yes').astype(float)
    return new_df


train_df = encode_binary_features(train_df)
test_df = encode_binary_features(test_df)

raw_features = [
    "AccountLength", "InternationalPlan", "VoiceMailPlan", "NumVoiceMailMessages",
    "DayMinutes", "DayCalls", "EveMinutes", "EveCalls", "NightMinutes",
    "NightCalls", "IntlMinutes", "IntlCalls", "CustomerServiceCalls"
]

# new features
def add_features(df):
    
    period_minutes = df["DayMinutes"] + df["EveMinutes"] + df["NightMinutes"]
    
    total_minutes = period_minutes + df["IntlMinutes"]
    
    total_calls = (
        df["DayCalls"] + df["EveCalls"] + df["NightCalls"] + df["IntlCalls"]
    )
    
    total_charge = (
        df["DayCharge"] + df["EveCharge"] + df["NightCharge"] + df["IntlCharge"]
    )
    
    return df.assign(
        # totals
        TotalCharge=total_charge,
        TotalMinutes=total_minutes,
        TotalCalls=total_calls,

        # service call thresholds
        ServiceCalls4Plus=(df["CustomerServiceCalls"] >= 4).astype(float),
        ServiceCalls8Plus=(df["CustomerServiceCalls"] >= 8).astype(float),

        # international usage of customers with plan
        IntlCallsWithPlan=df["IntlCalls"] * df.InternationalPlan,
        IntlMinutesWithPlan=df["IntlMinutes"] * df.InternationalPlan,

        # share of minutes by period
        DayMinutesShare=df["DayMinutes"].div(period_minutes).fillna(0.0),
        NightMinutesShare=df["NightMinutes"].div(period_minutes).fillna(0.0)
    )


extended_train_df = add_features(train_df)
extended_test_df = add_features(test_df)

new_columns = [
    name for name in extended_train_df.columns
    if name not in train_df.columns
]

all_columns = raw_features + new_columns

final_svm_features = [
    'NumVoiceMailMessages',
    'CustomerServiceCalls',
    'TotalCharge',
    'IntlCallsWithPlan',
    'IntlMinutesWithPlan'
]

X_train_xgb = extended_train_df[all_columns]
X_train_svm = extended_train_df[final_svm_features]
y_train = train_df["Churn"]

X_test_xgb = extended_test_df[all_columns]
X_test_svm = extended_test_df[final_svm_features]

###########################################
#            XGBoost pipeline             #
###########################################
nunique = X_train_xgb.nunique()
num_cols = nunique[nunique > 2].index.tolist()

# Create numerical pipeline
num_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.9))
])

# Standardize numerical features
preprocessor = ColumnTransformer(
    [("num", num_pipe, num_cols)],
    remainder="passthrough",
)

xgb_pipeline = Pipeline([
    ("prep", preprocessor),
    ("xgbclassifier", XGBClassifier()),
])

xgb_pipeline.fit(X_train_xgb, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test_xgb)

submission_xgb = pd.DataFrame(y_pred_xgb, index=X_test_xgb.index, columns=["Churn"])
submission_xgb.to_csv(data_folder / "xgboost_submission.csv", index=True)

###########################################
#                SVM model                #
###########################################

def laplacian(gamma):
    return lambda X, Y: laplacian_kernel(X, Y, gamma=gamma)

best_svm = SVC(kernel=laplacian(0.02), C=300)

best_svm.fit(X_train_svm, y_train)
y_pred_svm = best_svm.predict(X_test_svm)

submission_svm = pd.DataFrame(y_pred_svm, index=X_test_svm.index, columns=["Churn"])
submission_svm.to_csv(data_folder / "svm_submission.csv", index=True)

