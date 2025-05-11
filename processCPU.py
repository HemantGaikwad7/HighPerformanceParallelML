import multiprocessing
import time
import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.dataframe import from_pandas
from dask_ml.model_selection import train_test_split
from sklearn import preprocessing
from dask_ml.preprocessing import DummyEncoder
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def load_and_preprocess():
    print("\n[1/8] Starting data loading and preprocessing...")
    
    # Read Data
    try:
        print("[2/8] Reading CSV file...")
        data = pd.read_csv('onlinefraud.csv')
        print(f"  - Loaded {len(data)} records")
    except FileNotFoundError:
        print("ERROR: 'onlinefraud.csv' file not found")
        exit(1)
    
    # Drop unnecessary columns
    print("[3/8] Dropping unnecessary columns...")
    data = data.drop(['isFlaggedFraud','nameOrig','nameDest','step'], axis=1)
    print(f"  - Remaining columns: {list(data.columns)}")
    
    # Convert to Dask DataFrame
    print("[4/8] Converting to Dask DataFrame...")
    data_dd = from_pandas(data, npartitions=16)
    print(f"  - Created {data_dd.npartitions} partitions")
    
    # OneHotEncoder
    print("[5/8] Performing one-hot encoding...")
    encoder = DummyEncoder()
    data_dd = data_dd.categorize(['type'])
    data_dd = encoder.fit_transform(data_dd)
    print("  - Encoding completed")
    
    # Feature engineering
    print("[6/8] Engineering new features...")
    data_dd['orig_balance_diff'] = data_dd.oldbalanceOrg - data_dd.newbalanceOrig
    data_dd['dest_balance_diff'] = data_dd.oldbalanceDest - data_dd.newbalanceDest
    data_dd = data_dd.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest'], axis=1)
    data_dd = data_dd.drop(['type_PAYMENT','type_CASH_IN','type_DEBIT'], axis=1)
    print("  - Feature engineering complete")
    
    # Oversampling
    print("[7/8] Performing oversampling...")
    fraud_counts = data_dd['isFraud'].value_counts().compute()
    print(f"  - Original class distribution: {fraud_counts.to_dict()}")
    IsNotFraud = data_dd[data_dd['isFraud'] == 0]
    IsFraud = data_dd[data_dd['isFraud'] == 1]
    IsFraud_over = IsFraud.sample(frac=773.7, replace=True)
    over_balanced_data_dd = dd.concat([IsNotFraud, IsFraud_over], axis=0)
    print("  - Oversampling completed")
    
    # Prepare features
    print("[8/8] Final data preparation...")
    print("  - Converting to pandas DataFrame...")
    X = over_balanced_data_dd.drop('isFraud', axis=1).compute()
    y = over_balanced_data_dd['isFraud'].compute()
    
    # Split and scale
    print("  - Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=0,
        shuffle=True
    )
    
    print("  - Scaling features...")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    print("✅ Data preprocessing completed successfully")
    return X_train, X_test, y_train, y_test

def Model_Build(n_jobs, X_train, y_train):
    print(f"\n🏗️ Starting model building with {n_jobs} core(s)...")
    time1 = time.perf_counter()
    
    parameters = {
    'max_depth': [5, 15],          # Reduced from 3 to 2 values
    'learning_rate': [0.1],        # Fixed single good value
    'n_estimators': [100,200],         # Fixed single value
    'min_child_weight': [1,5],       # Fixed single value
    'subsample': [0.8],  # Reduced from 3 to 2 values
    }

    print("🔍 Initializing XGBoost classifier...")
    clf = XGBClassifier(
        tree_method='hist', 
        n_jobs=n_jobs,
        objective='binary:logistic',
        seed=1440
    )

    print("⚙️ Configuring GridSearchCV...")
    grid_result = GridSearchCV(
        clf, 
        param_grid=parameters, 
        scoring='f1', 
        cv=2, 
        n_jobs=n_jobs,
        error_score='raise',
        verbose=1  # Added for more detailed output
    )
    
    try:
        print("🔄 Starting grid search... (this may take a while)")
        grid_result.fit(X_train, y_train)
        print("🎉 Grid search completed successfully")
    except Exception as e:
        print(f"❌ Error during model fitting: {str(e)}")
        return

    print("\n📊 Best model results:")
    print(f"  - Best score ({n_jobs} cores): {grid_result.best_score_:.3f}")
    print("  - Best parameters:")
    for param, value in grid_result.best_params_.items():
        print(f"    {param}: {value}")
        
    time2 = time.perf_counter()
    print(f"\n⏱️ Total execution time: {time2 - time1:.2f} seconds")

if __name__ == '__main__':
    print("🚀 Starting fraud detection pipeline")
    print("="*50)
    
    # Load data once in main process
    print("\n📂 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    core_counts = [7]
    
    # CPU Benchmarking
    print("\n" + "="*50)
    print("💻 Beginning CPU benchmarking")
    print("="*50)
    
    for cores in core_counts:
        print(f"\n⚡ Testing with {cores} CPU core(s)")
        p = multiprocessing.Process(
            target=Model_Build,
            args=(cores, X_train, y_train)
        )
        p.start()
        p.join()
        print(f"✅ Completed test with {cores} core(s)")
    
    print("\n" + "="*50)
    print("🏁 All tests completed")
    print("="*50)
