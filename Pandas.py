import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold, cross_val_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

print("Step 1: Reading dataset...")
data = pd.read_csv('onlinefraud.csv')
print("Dataset loaded successfully.")

print("Step 2: Dropping unnecessary columns...")
data = data.drop(['isFlaggedFraud', 'nameOrig', 'nameDest', 'step'], axis=1)

print("Step 3: One-hot encoding 'type' feature...")
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
type_enc = enc.fit_transform(data['type'].values.reshape(-1, 1))
type_df = pd.DataFrame(type_enc, columns=enc.get_feature_names_out(['type']))
data = pd.concat([data.drop('type', axis=1), type_df], axis=1)

print("Dropping less relevant type columns...")
data = data.drop(['type_PAYMENT', 'type_CASH_IN', 'type_DEBIT'], axis=1)

print("Step 4: Creating new features...")
data['orig_balance_diff'] = data.oldbalanceOrg - data.newbalanceOrig
data['dest_balance_diff'] = data.oldbalanceDest - data.newbalanceDest

print("Dropping original balance columns...")
data = data.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], axis=1)

print("Step 5: Performing oversampling to balance classes...")
is_not_fraud, is_fraud = data[data['isFraud'] == 0], data[data['isFraud'] == 1]
is_fraud_over = is_fraud.sample(len(is_not_fraud), replace=True)
balanced_data = pd.concat([is_not_fraud, is_fraud_over])
print("Oversampling complete. Class balance achieved.")

print("Step 6: Splitting features and target...")
X = balanced_data.drop('isFraud', axis=1).values
y = balanced_data['isFraud'].values

print("Step 7: Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

print("Step 8: Feature scaling...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature scaling complete.")

# Common param grid
param_grid = {
    'max_depth': [5, 10],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 5],
    'subsample': [0.8, 0.95],
    'reg_alpha': [0, 1],
    'max_delta_step': [0, 1]
}

def build_model(cpu_count=None, gpu_id=None):
    mode = f"{cpu_count} CPU(s)" if gpu_id is None else f"GPU ID {gpu_id}"
    print(f"\n===== Training with {mode} =====")
    start_time = time.perf_counter()

    xgb_params = {
        'objective': 'binary:logistic',
        'seed': 42,
        'colsample_bytree': 0.8,
        'use_label_encoder': False
    }

    if gpu_id is not None:
        xgb_params.update({
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': gpu_id
        })
    else:
        xgb_params.update({
            'nthread': cpu_count,
            'tree_method': 'hist'
        })

    xgb = XGBClassifier(**xgb_params)

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=15,
        scoring='f1',
        cv=3,
        verbose=0,
        random_state=42,
        n_jobs=cpu_count if gpu_id is None else 1
    )

    search.fit(X_train, y_train)

    print("Best F1 score: %.4f" % search.best_score_)
    print("Best Parameters:")
    for key, val in search.best_params_.items():
        print(f"  {key}: {val}")

    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    results = cross_val_score(search.best_estimator_, X, y, cv=kfold, n_jobs=1)
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (results.mean() * 100, results.std() * 100))

    elapsed = time.perf_counter() - start_time
    print(f"Total Time Taken with {mode}: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)\n")

# Run for 1 to 8 CPUs
if __name__ == "__main__":
    print("\n=========== CPU MODE ===========")
    for cpu in range(1, 9):
        build_model(cpu_count=cpu)

    print("\n=========== GPU MODE ===========")
    for gpu in range(0, 8):
        try:
            build_model(gpu_id=gpu)
        except Exception as e:
            print(f"Skipping GPU {gpu}: {str(e)}")
