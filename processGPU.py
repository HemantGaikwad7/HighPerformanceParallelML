import multiprocessing
import time
import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.dataframe import from_pandas
from dask_ml.model_selection import train_test_split
from sklearn import preprocessing
from dask_ml.preprocessing import DummyEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def load_and_preprocess():
    print("\n[1/8] Starting data loading and preprocessing...")

    try:
        print("[2/8] Reading CSV file...")
        data = pd.read_csv('onlinefraud.csv')
        print(f"  - Loaded {len(data)} records")
    except FileNotFoundError:
        print("ERROR: 'olfr.csv' file not found")
        exit(1)

    print("[3/8] Dropping unnecessary columns...")
    data = data.drop(['isFlaggedFraud', 'nameOrig', 'nameDest', 'step'], axis=1)
    print(f"  - Remaining columns: {list(data.columns)}")

    print("[4/8] Converting to Dask DataFrame...")
    data_dd = from_pandas(data, npartitions=16)
    print(f"  - Created {data_dd.npartitions} partitions")

    print("[5/8] Performing one-hot encoding...")
    encoder = DummyEncoder()
    data_dd = data_dd.categorize(['type'])
    data_dd = encoder.fit_transform(data_dd)
    print("  - Encoding completed")

    print("[6/8] Engineering new features...")
    data_dd['orig_balance_diff'] = data_dd.oldbalanceOrg - data_dd.newbalanceOrig
    data_dd['dest_balance_diff'] = data_dd.oldbalanceDest - data_dd.newbalanceDest
    data_dd = data_dd.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], axis=1)
    data_dd = data_dd.drop(['type_PAYMENT', 'type_CASH_IN', 'type_DEBIT'], axis=1)
    print("  - Feature engineering complete")

    print("[7/8] Performing oversampling...")
    fraud_counts = data_dd['isFraud'].value_counts().compute()
    print(f"  - Original class distribution: {fraud_counts.to_dict()}")
    IsNotFraud = data_dd[data_dd['isFraud'] == 0]
    IsFraud = data_dd[data_dd['isFraud'] == 1]
    IsFraud_over = IsFraud.sample(frac=773.7, replace=True)
    over_balanced_data_dd = dd.concat([IsNotFraud, IsFraud_over], axis=0)
    print("  - Oversampling completed")

    print("[8/8] Final data preparation...")
    print("  - Converting to pandas DataFrame...")
    X = over_balanced_data_dd.drop('isFraud', axis=1).compute()
    y = over_balanced_data_dd['isFraud'].compute()

    print("  - Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, shuffle=True)

    print("  - Scaling features...")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print("✅ Data preprocessing completed successfully")
    return X_train, X_test, y_train, y_test

def Model_Build(n_jobs, X_train, y_train, use_gpu=False):
    print(f"\n🏗️ Starting model building with {n_jobs} core(s) on {'GPU' if use_gpu else 'CPU'}...")
    time1 = time.perf_counter()

    parameters = {
        'max_depth': [5, 10,25],
        'learning_rate': [0.1],
        'n_estimators': [100,200],
        'min_child_weight': [1.5],
        'subsample': [0.8],
    }

    print("🔍 Initializing XGBoost classifier...")
    clf = XGBClassifier(
        tree_method= 'hist',
        n_jobs=n_jobs,
        objective='binary:logistic',
        seed=1440,
        device='cuda' if use_gpu else 'cpu'
    )

    print("⚙️ Configuring GridSearchCV...")
    grid_result = GridSearchCV(
        clf,
        param_grid=parameters,
        scoring='f1',
        cv=3,
        n_jobs=n_jobs,
        error_score='raise',
        verbose=1
    )

    try:
        print("🔄 Starting grid search... (this may take a while)")
        grid_result.fit(X_train, y_train)
        print("🎉 Grid search completed successfully")
    except Exception as e:
        print(f"❌ Error during model fitting: {str(e)}")
        return

    print("\n📊 Best model results:")
    print(f"  - Best score ({n_jobs} cores, {'GPU' if use_gpu else 'CPU'}): {grid_result.best_score_:.3f}")
    for param, value in grid_result.best_params_.items():
        print(f"    {param}: {value}")

    time2 = time.perf_counter()
    print(f"\n⏱️ Total execution time: {time2 - time1:.2f} seconds")


def run_gpu_benchmark(X_train, y_train):
    # print(f"\n⚡ Testing with GPU")
    # Model_Build(4, X_train, y_train, True)
    print(f"\n⚡ Testing with GPU")
    p = multiprocessing.Process(
        target=Model_Build,
        args=(1, X_train, y_train, True)
    )
    p.start()
    p.join()
    print("✅ Completed GPU benchmark")

def process():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    core_counts = [2]  # You can change this to test with more cores

    print("\n" + "=" * 50)
    print("🖥️ Starting GPU Benchmark")
    print("=" * 50)
    run_gpu_benchmark(X_train, y_train)

    print("\n🏁 All tests completed")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    process()
