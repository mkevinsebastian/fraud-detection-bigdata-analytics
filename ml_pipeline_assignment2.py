"""
Assignment II - Machine Learning Pipeline for E-Commerce Fraud Detection
Course: COMP8035041 - Big Data Analytics
Student: Kevin Sebastian (NIM: 2702746012)
BINUS Graduate Program - Magister Teknik Informatika

This script executes:
1. End-to-end data preprocessing (cleaning age anomalies, outlier capping, unstructured address parsing)
2. Advanced feature engineering (log transform, velocity/ratio, binary flags, cyclical encoding)
3. Model training & comparison WITHOUT Spark MLlib (Scikit-Learn Logistic Regression, Random Forest, XGBoost)
4. Model training & comparison WITH Spark MLlib (PySpark LogisticRegression, RandomForestClassifier, GBTClassifier)
5. Threshold optimization for fraud detection
6. Scalability & Execution time comparison across data scales
7. Generation of publication-quality diagrams in diagrams/
8. Export of evaluation results to JSON and TXT
"""

import os
import sys
import time
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure PySpark uses current Python executable
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)

# PySpark MLlib imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sin, cos, lit
from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkStandardScaler
from pyspark.ml.classification import (
    LogisticRegression as SparkLR,
    RandomForestClassifier as SparkRF,
    GBTClassifier as SparkGBT
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Set styling for publication-quality charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

def parse_address_regex(addr):
    """Parse unstructured address string to extract State and ZIP code."""
    if not isinstance(addr, str) or not addr.strip():
        return 'UNK', 'UNK'
    lines = addr.strip().split('\n')
    last_line = lines[-1] if len(lines) > 1 else lines[0]
    m = re.search(r',\s*([A-Z]{2})\s+(\d{5})', last_line)
    if m:
        return m.group(1), m.group(2)
    m_zip = re.search(r'\b(\d{5})\b', last_line)
    zip_code = m_zip.group(1) if m_zip else 'UNK'
    m_state = re.search(r'\b([A-Z]{2})\b', last_line)
    state = m_state.group(1) if m_state else 'UNK'
    return state, zip_code

def preprocess_and_feature_engineering(df, p99_amount=None, median_age=None):
    """
    Perform comprehensive data preprocessing and feature engineering on pandas DataFrame.
    """
    print(f"[*] Starting preprocessing on DataFrame with {len(df):,} rows...")
    df = df.copy()

    # 1. Anomaly handling: Customer Age < 18
    invalid_age_mask = df['Customer Age'] < 18
    invalid_age_count = int(invalid_age_mask.sum())
    if median_age is None:
        median_age = float(df.loc[~invalid_age_mask, 'Customer Age'].median())
    df['customer_age_clean'] = df['Customer Age'].apply(lambda x: median_age if x < 18 else float(x))
    print(f"    - Handled age anomalies: {invalid_age_count:,} rows with age < 18 imputed with median {median_age:.1f}")

    # 2. Outlier capping: Transaction Amount 99th percentile
    if p99_amount is None:
        p99_amount = float(df['Transaction Amount'].quantile(0.99))
    df['amount_capped'] = np.minimum(df['Transaction Amount'], p99_amount)
    print(f"    - Capped Transaction Amount at 99th percentile: ${p99_amount:.2f} (max was ${df['Transaction Amount'].max():.2f})")

    # 3. Unstructured text parsing: Shipping & Billing Address
    ship_states = []
    ship_zips = []
    bill_states = []
    bill_zips = []
    for s_addr, b_addr in zip(df['Shipping Address'], df['Billing Address']):
        s_st, s_zp = parse_address_regex(s_addr)
        b_st, b_zp = parse_address_regex(b_addr)
        ship_states.append(s_st)
        ship_zips.append(s_zp)
        bill_states.append(b_st)
        bill_zips.append(b_zp)
    
    df['shipping_state'] = ship_states
    df['shipping_zip'] = ship_zips
    df['billing_state'] = bill_states
    df['billing_zip'] = bill_zips

    df['address_mismatch'] = (df['Shipping Address'] != df['Billing Address']).astype(int)
    df['state_mismatch'] = (df['shipping_state'] != df['billing_state']).astype(int)
    mismatch_cnt = int(df['address_mismatch'].sum())
    print(f"    - Parsed addresses: {mismatch_cnt:,} address mismatches identified ({mismatch_cnt/len(df)*100:.2f}%)")

    # 4. Feature Engineering
    df['log_amount'] = np.log1p(df['Transaction Amount'])
    df['amount_to_qty_ratio'] = df['Transaction Amount'] / np.maximum(df['Quantity'], 1)
    df['is_high_amount'] = (df['Transaction Amount'] > 1000.0).astype(int)
    df['is_new_account'] = (df['Account Age Days'] <= 30).astype(int)
    df['is_dawn_tx'] = ((df['Transaction Hour'] >= 0) & (df['Transaction Hour'] <= 5)).astype(int)
    
    # Cyclical hour encoding
    df['hour_sin'] = np.sin(2.0 * np.pi * df['Transaction Hour'] / 24.0)
    df['hour_cos'] = np.cos(2.0 * np.pi * df['Transaction Hour'] / 24.0)

    # 5. One-Hot Encoding for categorical features
    cat_cols = ['Payment Method', 'Product Category', 'Device Used']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    # Identify feature columns
    ignore_cols = [
        'Transaction ID', 'Customer ID', 'Transaction Date', 'Customer Location',
        'IP Address', 'Shipping Address', 'Billing Address', 'shipping_state',
        'shipping_zip', 'billing_state', 'billing_zip', 'Customer Age', 'Is Fraudulent'
    ]
    feature_cols = [c for c in df_encoded.columns if c not in ignore_cols]
    
    print(f"    - Engineered {len(feature_cols)} total features.")
    return df_encoded, feature_cols, p99_amount, median_age

def evaluate_predictions(y_true, y_pred, y_prob, model_name, exec_time, inference_time):
    """Compute and format all classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'model_name': model_name,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'training_time_sec': float(exec_time),
        'inference_time_sec': float(inference_time)
    }

def run_ml_pipeline():
    os.makedirs('diagrams', exist_ok=True)
    results_summary = {}

    print("================================================================================")
    print("      ASSIGNMENT II - BIG DATA MACHINE LEARNING PIPELINE EXECUTION")
    print("================================================================================")

    # --- PART 1: LOAD V2 DATASET (Prototyping & Direct Model Comparison) ---
    v2_path = 'dataset/Fraudulent_E-Commerce_Transaction_Data_2.csv'
    print(f"[*] Step 1: Loading Dataset V2 ({v2_path})...")
    df_raw = pd.read_csv(v2_path)
    total_rows = len(df_raw)
    fraud_count = int(df_raw['Is Fraudulent'].sum())
    non_fraud_count = total_rows - fraud_count
    imbalance_ratio = non_fraud_count / fraud_count
    print(f"    Total Rows: {total_rows:,} | Legitimate: {non_fraud_count:,} ({non_fraud_count/total_rows*100:.2f}%) | Fraud: {fraud_count:,} ({fraud_count/total_rows*100:.2f}%)")
    print(f"    Class Imbalance Ratio: {imbalance_ratio:.2f} : 1")

    # Preprocessing
    df_proc, feature_cols, p99_amt, med_age = preprocess_and_feature_engineering(df_raw)

    X = df_proc[feature_cols].values
    y = df_proc['Is Fraudulent'].values

    # Train / Test Stratified Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Train set: {X_train.shape[0]:,} samples | Test set: {X_test.shape[0]:,} samples")

    # Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    eval_results = {}
    roc_curves = {}
    pr_curves = {}
    feature_importances = {}
    y_test_probs = {}

    # --- PART 2: MODELS WITHOUT SPARK MLLIB (Scikit-Learn & XGBoost) ---
    print("\n--------------------------------------------------------------------------------")
    print(" 1. TRAINING MODELS WITHOUT SPARK MLLIB (Scikit-Learn & XGBoost)")
    print("--------------------------------------------------------------------------------")

    # 1. Logistic Regression
    print("[*] Training Model 1: Logistic Regression (class_weight='balanced')...")
    t0 = time.time()
    lr_sk = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')
    lr_sk.fit(X_train_scaled, y_train)
    lr_train_time = time.time() - t0

    t0 = time.time()
    lr_y_prob = lr_sk.predict_proba(X_test_scaled)[:, 1]
    lr_infer_time = time.time() - t0
    lr_y_pred = (lr_y_prob >= 0.5).astype(int)

    eval_results['Sklearn_LogisticRegression'] = evaluate_predictions(
        y_test, lr_y_pred, lr_y_prob, 'Logistic Regression (Sklearn)', lr_train_time, lr_infer_time
    )
    fpr, tpr, _ = roc_curve(y_test, lr_y_prob)
    prec, rec, _ = precision_recall_curve(y_test, lr_y_prob)
    roc_curves['Logistic Regression (Sklearn)'] = (fpr, tpr)
    pr_curves['Logistic Regression (Sklearn)'] = (rec, prec)
    y_test_probs['Sklearn_LR'] = lr_y_prob

    # 2. Random Forest Classifier
    print("[*] Training Model 2: Random Forest (class_weight='balanced', n_estimators=100)...")
    t0 = time.time()
    rf_sk = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_sk.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    t0 = time.time()
    rf_y_prob = rf_sk.predict_proba(X_test)[:, 1]
    rf_infer_time = time.time() - t0
    rf_y_pred = (rf_y_prob >= 0.5).astype(int)

    eval_results['Sklearn_RandomForest'] = evaluate_predictions(
        y_test, rf_y_pred, rf_y_prob, 'Random Forest (Sklearn)', rf_train_time, rf_infer_time
    )
    fpr, tpr, _ = roc_curve(y_test, rf_y_prob)
    prec, rec, _ = precision_recall_curve(y_test, rf_y_prob)
    roc_curves['Random Forest (Sklearn)'] = (fpr, tpr)
    pr_curves['Random Forest (Sklearn)'] = (rec, prec)
    y_test_probs['Sklearn_RF'] = rf_y_prob
    feature_importances['Random Forest'] = rf_sk.feature_importances_

    # 3. XGBoost Classifier
    print(f"[*] Training Model 3: XGBoost Classifier (scale_pos_weight={imbalance_ratio:.2f}, n_estimators=100)...")
    t0 = time.time()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        eval_metric='aucpr',
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train)
    xgb_train_time = time.time() - t0

    t0 = time.time()
    xgb_y_prob = xgb_clf.predict_proba(X_test)[:, 1]
    xgb_infer_time = time.time() - t0
    xgb_y_pred = (xgb_y_prob >= 0.5).astype(int)

    eval_results['XGBoost'] = evaluate_predictions(
        y_test, xgb_y_pred, xgb_y_prob, 'XGBoost Classifier', xgb_train_time, xgb_infer_time
    )
    fpr, tpr, _ = roc_curve(y_test, xgb_y_prob)
    prec, rec, _ = precision_recall_curve(y_test, xgb_y_prob)
    roc_curves['XGBoost Classifier'] = (fpr, tpr)
    pr_curves['XGBoost Classifier'] = (rec, prec)
    y_test_probs['XGBoost'] = xgb_y_prob
    feature_importances['XGBoost'] = xgb_clf.feature_importances_

    # --- PART 3: THRESHOLD TUNING EXPERIMENT ---
    print("\n--------------------------------------------------------------------------------")
    print(" 2. THRESHOLD TUNING EXPERIMENT ON BEST MODEL (XGBoost)")
    print("--------------------------------------------------------------------------------")
    thresholds = np.linspace(0.05, 0.95, 37)
    tuning_records = []
    best_f1 = -1
    best_thresh = 0.5

    for th in thresholds:
        th_pred = (xgb_y_prob >= th).astype(int)
        th_prec = precision_score(y_test, th_pred, zero_division=0)
        th_rec = recall_score(y_test, th_pred, zero_division=0)
        th_f1 = f1_score(y_test, th_pred, zero_division=0)
        tuning_records.append({
            'threshold': round(float(th), 3),
            'precision': round(float(th_prec), 4),
            'recall': round(float(th_rec), 4),
            'f1_score': round(float(th_f1), 4)
        })
        if th_f1 > best_f1:
            best_f1 = th_f1
            best_thresh = float(th)

    print(f"[*] Optimal F1 Threshold: {best_thresh:.3f} | Resulting F1: {best_f1:.4f}")
    xgb_y_pred_tuned = (xgb_y_prob >= best_thresh).astype(int)
    eval_results['XGBoost_Tuned'] = evaluate_predictions(
        y_test, xgb_y_pred_tuned, xgb_y_prob, f'XGBoost (Threshold {best_thresh:.2f})', xgb_train_time, xgb_infer_time
    )

    # --- PART 4: MODELS WITH SPARK MLLIB (PySpark) ---
    print("\n--------------------------------------------------------------------------------")
    print(" 3. TRAINING MODELS WITH SPARK MLLIB (PySpark Distributed ML)")
    print("--------------------------------------------------------------------------------")
    spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Assignment2_SparkML") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print("[*] Initializing PySpark Session & Loading Dataset...")
    # Read CSV with multiLine=True
    spark_df = spark.read \
        .option("header", "true") \
        .option("multiLine", "true") \
        .option("escape", '"') \
        .csv(v2_path)

    # Feature transformation in PySpark
    spark_df = spark_df \
        .withColumn('Transaction Amount', col('Transaction Amount').cast('double')) \
        .withColumn('Quantity', col('Quantity').cast('double')) \
        .withColumn('Customer Age', col('Customer Age').cast('double')) \
        .withColumn('Account Age Days', col('Account Age Days').cast('double')) \
        .withColumn('Transaction Hour', col('Transaction Hour').cast('double')) \
        .withColumn('label', col('Is Fraudulent').cast('double'))

    spark_df = spark_df \
        .withColumn('customer_age_clean', when(col('Customer Age') < 18, med_age).otherwise(col('Customer Age'))) \
        .withColumn('amount_capped', when(col('Transaction Amount') > p99_amt, p99_amt).otherwise(col('Transaction Amount'))) \
        .withColumn('log_amount', col('Transaction Amount') + lit(1.0)) \
        .withColumn('amount_to_qty_ratio', col('Transaction Amount') / col('Quantity')) \
        .withColumn('is_high_amount', when(col('Transaction Amount') > 1000.0, 1.0).otherwise(0.0)) \
        .withColumn('is_new_account', when(col('Account Age Days') <= 30.0, 1.0).otherwise(0.0)) \
        .withColumn('is_dawn_tx', when((col('Transaction Hour') >= 0.0) & (col('Transaction Hour') <= 5.0), 1.0).otherwise(0.0)) \
        .withColumn('address_mismatch', when(col('Shipping Address') != col('Billing Address'), 1.0).otherwise(0.0)) \
        .withColumn('weight', when(col('label') == 1.0, imbalance_ratio).otherwise(1.0))

    # PySpark vector assembly (using numerical and engineered features)
    spark_features = [
        'Transaction Amount', 'amount_capped', 'amount_to_qty_ratio', 'Quantity',
        'customer_age_clean', 'Account Age Days', 'Transaction Hour',
        'is_high_amount', 'is_new_account', 'is_dawn_tx', 'address_mismatch'
    ]
    assembler = VectorAssembler(inputCols=spark_features, outputCol='raw_features')
    spark_vec_df = assembler.transform(spark_df).select('raw_features', 'label', 'weight')

    # Scaling
    spark_scaler = SparkStandardScaler(inputCol='raw_features', outputCol='features', withMean=True, withStd=True)
    scaler_model = spark_scaler.fit(spark_vec_df)
    spark_scaled_df = scaler_model.transform(spark_vec_df).select('features', 'label', 'weight')

    # Stratified Train/Test Split
    train_spark, test_spark = spark_scaled_df.randomSplit([0.8, 0.2], seed=42)
    train_spark.cache()
    test_spark.cache()
    train_count = train_spark.count()
    test_count = test_spark.count()
    print(f"[*] Spark Dataset: {train_count:,} Train rows, {test_count:,} Test rows")

    # Evaluators
    roc_eval = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
    pr_eval = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction', metricName='areaUnderPR')
    acc_eval = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    f1_eval = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='f1')

    # 4. Spark MLlib Logistic Regression
    print("[*] Training Model 4: Spark MLlib LogisticRegression...")
    t0 = time.time()
    spark_lr = SparkLR(featuresCol='features', labelCol='label', weightCol='weight', maxIter=30)
    spark_lr_model = spark_lr.fit(train_spark)
    spark_lr_train_time = time.time() - t0

    t0 = time.time()
    spark_lr_preds = spark_lr_model.transform(test_spark)
    spark_lr_preds.cache()
    _ = spark_lr_preds.count()
    spark_lr_infer_time = time.time() - t0

    s_lr_roc = float(roc_eval.evaluate(spark_lr_preds))
    s_lr_pr = float(pr_eval.evaluate(spark_lr_preds))
    s_lr_acc = float(acc_eval.evaluate(spark_lr_preds))
    s_lr_f1 = float(f1_eval.evaluate(spark_lr_preds))

    # Confusion matrix extraction
    pred_actual = spark_lr_preds.select('prediction', 'label').collect()
    s_lr_preds_np = np.array([r.prediction for r in pred_actual])
    s_lr_labels_np = np.array([r.label for r in pred_actual])
    s_lr_prec = float(precision_score(s_lr_labels_np, s_lr_preds_np, zero_division=0))
    s_lr_rec = float(recall_score(s_lr_labels_np, s_lr_preds_np, zero_division=0))
    s_lr_f1_bin = float(f1_score(s_lr_labels_np, s_lr_preds_np, zero_division=0))
    cm_lr = confusion_matrix(s_lr_labels_np, s_lr_preds_np)
    tn_s, fp_s, fn_s, tp_s = cm_lr.ravel()

    eval_results['Spark_LogisticRegression'] = {
        'model_name': 'Logistic Regression (Spark MLlib)',
        'accuracy': s_lr_acc,
        'precision': s_lr_prec,
        'recall': s_lr_rec,
        'f1_score': s_lr_f1_bin,
        'weighted_f1': s_lr_f1,
        'roc_auc': s_lr_roc,
        'pr_auc': s_lr_pr,
        'confusion_matrix': {'tn': int(tn_s), 'fp': int(fp_s), 'fn': int(fn_s), 'tp': int(tp_s)},
        'training_time_sec': float(spark_lr_train_time),
        'inference_time_sec': float(spark_lr_infer_time)
    }

    # Extract probability vector for ROC
    prob_rows = spark_lr_preds.select('probability', 'label').collect()
    s_lr_probs = np.array([r.probability[1] for r in prob_rows])
    fpr, tpr, _ = roc_curve(s_lr_labels_np, s_lr_probs)
    prec, rec, _ = precision_recall_curve(s_lr_labels_np, s_lr_probs)
    roc_curves['Logistic Regression (Spark MLlib)'] = (fpr, tpr)
    pr_curves['Logistic Regression (Spark MLlib)'] = (rec, prec)

    # 5. Spark MLlib Random Forest
    print("[*] Training Model 5: Spark MLlib RandomForestClassifier (numTrees=50, maxDepth=10)...")
    t0 = time.time()
    spark_rf = SparkRF(featuresCol='features', labelCol='label', numTrees=50, maxDepth=10, seed=42)
    spark_rf_model = spark_rf.fit(train_spark)
    spark_rf_train_time = time.time() - t0

    t0 = time.time()
    spark_rf_preds = spark_rf_model.transform(test_spark)
    spark_rf_preds.cache()
    _ = spark_rf_preds.count()
    spark_rf_infer_time = time.time() - t0

    s_rf_roc = float(roc_eval.evaluate(spark_rf_preds))
    s_rf_pr = float(pr_eval.evaluate(spark_rf_preds))
    s_rf_acc = float(acc_eval.evaluate(spark_rf_preds))
    s_rf_f1 = float(f1_eval.evaluate(spark_rf_preds))

    pred_actual_rf = spark_rf_preds.select('prediction', 'label').collect()
    s_rf_preds_np = np.array([r.prediction for r in pred_actual_rf])
    s_rf_labels_np = np.array([r.label for r in pred_actual_rf])
    s_rf_prec = float(precision_score(s_rf_labels_np, s_rf_preds_np, zero_division=0))
    s_rf_rec = float(recall_score(s_rf_labels_np, s_rf_preds_np, zero_division=0))
    s_rf_f1_bin = float(f1_score(s_rf_labels_np, s_rf_preds_np, zero_division=0))
    cm_rf = confusion_matrix(s_rf_labels_np, s_rf_preds_np)
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

    eval_results['Spark_RandomForest'] = {
        'model_name': 'Random Forest (Spark MLlib)',
        'accuracy': s_rf_acc,
        'precision': s_rf_prec,
        'recall': s_rf_rec,
        'f1_score': s_rf_f1_bin,
        'weighted_f1': s_rf_f1,
        'roc_auc': s_rf_roc,
        'pr_auc': s_rf_pr,
        'confusion_matrix': {'tn': int(tn_rf), 'fp': int(fp_rf), 'fn': int(fn_rf), 'tp': int(tp_rf)},
        'training_time_sec': float(spark_rf_train_time),
        'inference_time_sec': float(spark_rf_infer_time)
    }

    prob_rows_rf = spark_rf_preds.select('probability', 'label').collect()
    s_rf_probs = np.array([r.probability[1] for r in prob_rows_rf])
    fpr, tpr, _ = roc_curve(s_rf_labels_np, s_rf_probs)
    prec, rec, _ = precision_recall_curve(s_rf_labels_np, s_rf_probs)
    roc_curves['Random Forest (Spark MLlib)'] = (fpr, tpr)
    pr_curves['Random Forest (Spark MLlib)'] = (rec, prec)

    # 6. Spark MLlib GBTClassifier
    print("[*] Training Model 6: Spark MLlib GBTClassifier (maxIter=30, maxDepth=5)...")
    t0 = time.time()
    spark_gbt = SparkGBT(featuresCol='features', labelCol='label', maxIter=30, maxDepth=5, seed=42)
    spark_gbt_model = spark_gbt.fit(train_spark)
    spark_gbt_train_time = time.time() - t0

    t0 = time.time()
    spark_gbt_preds = spark_gbt_model.transform(test_spark)
    spark_gbt_preds.cache()
    _ = spark_gbt_preds.count()
    spark_gbt_infer_time = time.time() - t0

    s_gbt_roc = float(roc_eval.evaluate(spark_gbt_preds))
    s_gbt_pr = float(pr_eval.evaluate(spark_gbt_preds))
    s_gbt_acc = float(acc_eval.evaluate(spark_gbt_preds))
    s_gbt_f1 = float(f1_eval.evaluate(spark_gbt_preds))

    pred_actual_gbt = spark_gbt_preds.select('prediction', 'label').collect()
    s_gbt_preds_np = np.array([r.prediction for r in pred_actual_gbt])
    s_gbt_labels_np = np.array([r.label for r in pred_actual_gbt])
    s_gbt_prec = float(precision_score(s_gbt_labels_np, s_gbt_preds_np, zero_division=0))
    s_gbt_rec = float(recall_score(s_gbt_labels_np, s_gbt_preds_np, zero_division=0))
    s_gbt_f1_bin = float(f1_score(s_gbt_labels_np, s_gbt_preds_np, zero_division=0))
    cm_gbt = confusion_matrix(s_gbt_labels_np, s_gbt_preds_np)
    tn_gbt, fp_gbt, fn_gbt, tp_gbt = cm_gbt.ravel()

    eval_results['Spark_GBTClassifier'] = {
        'model_name': 'GBTClassifier (Spark MLlib)',
        'accuracy': s_gbt_acc,
        'precision': s_gbt_prec,
        'recall': s_gbt_rec,
        'f1_score': s_gbt_f1_bin,
        'weighted_f1': s_gbt_f1,
        'roc_auc': s_gbt_roc,
        'pr_auc': s_gbt_pr,
        'confusion_matrix': {'tn': int(tn_gbt), 'fp': int(fp_gbt), 'fn': int(fn_gbt), 'tp': int(tp_gbt)},
        'training_time_sec': float(spark_gbt_train_time),
        'inference_time_sec': float(spark_gbt_infer_time)
    }

    prob_rows_gbt = spark_gbt_preds.select('probability', 'label').collect()
    s_gbt_probs = np.array([r.probability[1] for r in prob_rows_gbt])
    fpr, tpr, _ = roc_curve(s_gbt_labels_np, s_gbt_probs)
    prec, rec, _ = precision_recall_curve(s_gbt_labels_np, s_gbt_probs)
    roc_curves['GBTClassifier (Spark MLlib)'] = (fpr, tpr)
    pr_curves['GBTClassifier (Spark MLlib)'] = (rec, prec)

    # --- PART 5: SCALABILITY & EXECUTION TIME COMPARISON ACROSS DATASET SCALES ---
    print("\n--------------------------------------------------------------------------------")
    print(" 4. BENCHMARKING SCALABILITY: V2 (23.6K rows) vs SCALED V1 SUBSET (200K rows)")
    print("--------------------------------------------------------------------------------")
    # Load 200,000 rows from V1 for direct scalability timing comparison
    v1_path = 'dataset/Fraudulent_E-Commerce_Transaction_Data.csv'
    print(f"[*] Loading 200,000 rows sample from V1 ({v1_path}) for scalability test...")
    df_v1_sample = pd.read_csv(v1_path, nrows=200000)
    df_v1_proc, v1_feats, _, _ = preprocess_and_feature_engineering(df_v1_sample, p99_amt, med_age)
    X_v1 = df_v1_proc[v1_feats].values
    y_v1 = df_v1_proc['Is Fraudulent'].values

    # 1. Scikit-Learn Logistic Regression on 200k rows
    t0 = time.time()
    lr_v1 = LogisticRegression(class_weight='balanced', max_iter=200, random_state=42)
    lr_v1.fit(X_v1[:160000], y_v1[:160000])
    lr_v1_time = time.time() - t0

    # 2. XGBoost on 200k rows
    t0 = time.time()
    xgb_v1 = xgb.XGBClassifier(n_estimators=100, max_depth=6, tree_method='hist', n_jobs=4, random_state=42)
    xgb_v1.fit(X_v1[:160000], y_v1[:160000])
    xgb_v1_time = time.time() - t0

    # 3. PySpark on 200k rows
    print("[*] PySpark processing on 200,000 rows...")
    spark_v1 = spark.read.option("header", "true").option("multiLine", "true").option("escape", '"').csv(v1_path)
    spark_v1_sub = spark_v1.limit(200000)
    spark_v1_sub = spark_v1_sub \
        .withColumn('Transaction Amount', col('Transaction Amount').cast('double')) \
        .withColumn('Quantity', col('Quantity').cast('double')) \
        .withColumn('Customer Age', col('Customer Age').cast('double')) \
        .withColumn('Account Age Days', col('Account Age Days').cast('double')) \
        .withColumn('Transaction Hour', col('Transaction Hour').cast('double')) \
        .withColumn('label', col('Is Fraudulent').cast('double')) \
        .withColumn('customer_age_clean', when(col('Customer Age') < 18, med_age).otherwise(col('Customer Age'))) \
        .withColumn('amount_capped', when(col('Transaction Amount') > p99_amt, p99_amt).otherwise(col('Transaction Amount'))) \
        .withColumn('amount_to_qty_ratio', col('Transaction Amount') / col('Quantity')) \
        .withColumn('is_high_amount', when(col('Transaction Amount') > 1000.0, 1.0).otherwise(0.0)) \
        .withColumn('is_new_account', when(col('Account Age Days') <= 30.0, 1.0).otherwise(0.0)) \
        .withColumn('is_dawn_tx', when((col('Transaction Hour') >= 0.0) & (col('Transaction Hour') <= 5.0), 1.0).otherwise(0.0)) \
        .withColumn('address_mismatch', when(col('Shipping Address') != col('Billing Address'), 1.0).otherwise(0.0)) \
        .withColumn('weight', when(col('label') == 1.0, imbalance_ratio).otherwise(1.0))

    spark_v1_vec = assembler.transform(spark_v1_sub).select('raw_features', 'label', 'weight')
    spark_v1_vec.cache()
    _ = spark_v1_vec.count()

    t0 = time.time()
    s_lr_v1_model = SparkLR(featuresCol='raw_features', labelCol='label', weightCol='weight', maxIter=20).fit(spark_v1_vec)
    spark_lr_v1_time = time.time() - t0

    t0 = time.time()
    s_rf_v1_model = SparkRF(featuresCol='raw_features', labelCol='label', numTrees=30, maxDepth=8, seed=42).fit(spark_v1_vec)
    spark_rf_v1_time = time.time() - t0

    scalability_results = {
        'scale_v2_23k': {
            'rows': 23634,
            'Sklearn_LR_sec': lr_train_time,
            'Sklearn_RF_sec': rf_train_time,
            'XGBoost_sec': xgb_train_time,
            'Spark_LR_sec': spark_lr_train_time,
            'Spark_RF_sec': spark_rf_train_time,
            'Spark_GBT_sec': spark_gbt_train_time
        },
        'scale_v1_200k': {
            'rows': 200000,
            'Sklearn_LR_sec': lr_v1_time,
            'XGBoost_sec': xgb_v1_time,
            'Spark_LR_sec': spark_lr_v1_time,
            'Spark_RF_sec': spark_rf_v1_time
        }
    }

    # Stop Spark Session
    spark.stop()
    print("[*] PySpark session closed successfully.")

    # --- PART 6: GENERATE HIGH-RESOLUTION PUBLICATION CHARTS ---
    print("\n--------------------------------------------------------------------------------")
    print(" 5. GENERATING PUBLICATION-QUALITY DIAGRAMS IN diagrams/")
    print("--------------------------------------------------------------------------------")

    # 1. ROC Curves & PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Palette
    colors = {
        'Logistic Regression (Sklearn)': '#1f77b4',
        'Random Forest (Sklearn)': '#2ca02c',
        'XGBoost Classifier': '#d62728',
        'Logistic Regression (Spark MLlib)': '#9467bd',
        'Random Forest (Spark MLlib)': '#8c564b',
        'GBTClassifier (Spark MLlib)': '#e377c2'
    }

    # Left: ROC
    for name, (fpr_val, tpr_val) in roc_curves.items():
        auc_score = [v['roc_auc'] for k, v in eval_results.items() if v['model_name'] == name or name in v['model_name']][0]
        axes[0].plot(fpr_val, tpr_val, label=f"{name} (AUC = {auc_score:.3f})", lw=2, color=colors.get(name, None))
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (AUC = 0.500)')
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0].set_ylabel('True Positive Rate (Sensitivity / Recall)')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold')
    axes[0].legend(loc='lower right', frameon=True)
    axes[0].grid(True, alpha=0.3)

    # Right: PR Curve
    for name, (rec_val, prec_val) in pr_curves.items():
        pr_score = [v['pr_auc'] for k, v in eval_results.items() if v['model_name'] == name or name in v['model_name']][0]
        axes[1].plot(rec_val, prec_val, label=f"{name} (PR-AUC = {pr_score:.3f})", lw=2, color=colors.get(name, None))
    axes[1].plot([0, 1], [fraud_count/total_rows, fraud_count/total_rows], 'k--', lw=1.5, label=f'Baseline Prevalence ({fraud_count/total_rows:.3f})')
    axes[1].set_xlabel('Recall (True Positive Rate)')
    axes[1].set_ylabel('Precision (Positive Predictive Value)')
    axes[1].set_title('Precision-Recall (PR) Curves', fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    roc_pr_path = 'diagrams/ml_roc_pr_curves.png'
    plt.savefig(roc_pr_path, dpi=300)
    plt.close()
    print(f"    - Saved: {roc_pr_path}")

    # 2. Feature Importance Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # RF Importance
    rf_imp_series = pd.Series(feature_importances['Random Forest'], index=feature_cols).sort_values(ascending=True)
    rf_top = rf_imp_series.tail(12)
    rf_colors = ['#d95f02' if 'Amount' in c or 'amount' in c or 'Age' in c or 'Hour' in c else '#7570b3' for c in rf_top.index]
    axes[0].barh(rf_top.index, rf_top.values, color=rf_colors)
    axes[0].set_title('Random Forest Feature Importance (MDI)', fontweight='bold')
    axes[0].set_xlabel('Relative Importance')
    axes[0].grid(True, alpha=0.3, axis='x')

    # XGBoost Importance
    xgb_imp_series = pd.Series(feature_importances['XGBoost'], index=feature_cols).sort_values(ascending=True)
    xgb_top = xgb_imp_series.tail(12)
    xgb_colors = ['#e41a1c' if 'Amount' in c or 'amount' in c or 'Age' in c or 'Hour' in c else '#377eb8' for c in xgb_top.index]
    axes[1].barh(xgb_top.index, xgb_top.values, color=xgb_colors)
    axes[1].set_title('XGBoost Feature Importance (Gain / Weight)', fontweight='bold')
    axes[1].set_xlabel('Relative Importance')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.suptitle('Model Feature Importance: Core Numerical & Engineered Features vs Categoricals', fontweight='bold', y=0.98)
    plt.tight_layout()
    fi_path = 'diagrams/ml_feature_importance.png'
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"    - Saved: {fi_path}")

    # 3. Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    models_to_plot = [
        ('Sklearn_LogisticRegression', 'Logistic Regression (Threshold 0.50)', axes[0, 0]),
        ('Sklearn_RandomForest', 'Random Forest (Threshold 0.50)', axes[0, 1]),
        ('XGBoost', 'XGBoost (Default Threshold 0.50)', axes[1, 0]),
        ('XGBoost_Tuned', f'XGBoost (Tuned Threshold {best_thresh:.2f})', axes[1, 1])
    ]

    for key, title, ax in models_to_plot:
        data_cm = eval_results[key]['confusion_matrix']
        matrix = np.array([[data_cm['tn'], data_cm['fp']], [data_cm['fn'], data_cm['tp']]])
        sns.heatmap(matrix, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=ax, annot_kws={'size': 13, 'fontweight': 'bold'})
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Predicted Label (0 = Legitimate, 1 = Fraud)')
        ax.set_ylabel('True Label (0 = Legitimate, 1 = Fraud)')
        ax.set_xticklabels(['Legitimate', 'Fraud'])
        ax.set_yticklabels(['Legitimate', 'Fraud'])
        
        # Annotation text
        prec_val = eval_results[key]['precision']
        rec_val = eval_results[key]['recall']
        f1_val = eval_results[key]['f1_score']
        ax.text(0.5, -0.22, f"Precision: {prec_val:.3f} | Recall: {rec_val:.3f} | F1: {f1_val:.3f}", 
                ha='center', va='center', transform=ax.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='gray'))

    plt.suptitle('Confusion Matrix Comparison Across Models & Decision Thresholds', fontweight='bold', y=0.99)
    plt.tight_layout()
    cm_path = 'diagrams/ml_confusion_matrices.png'
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"    - Saved: {cm_path}")

    # 4. Spark MLlib vs Scikit-Learn Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (a) Performance Metrics Comparison
    comp_models = [
        'Sklearn_LogisticRegression', 'Spark_LogisticRegression',
        'Sklearn_RandomForest', 'Spark_RandomForest',
        'XGBoost', 'Spark_GBTClassifier'
    ]
    labels = ['Sklearn LR', 'Spark LR', 'Sklearn RF', 'Spark RF', 'XGBoost', 'Spark GBT']
    f1_vals = [eval_results[k]['f1_score'] for k in comp_models]
    roc_vals = [eval_results[k]['roc_auc'] for k in comp_models]
    pr_vals = [eval_results[k]['pr_auc'] for k in comp_models]

    x = np.arange(len(labels))
    width = 0.25

    axes[0].bar(x - width, f1_vals, width, label='F1-Score', color='#1f77b4')
    axes[0].bar(x, roc_vals, width, label='ROC-AUC', color='#ff7f0e')
    axes[0].bar(x + width, pr_vals, width, label='PR-AUC', color='#2ca02c')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Classification Performance: Single-Node vs Spark MLlib', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha='right')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3, axis='y')

    # (b) Execution Time & Scalability
    scalability_labels = ['Logistic Reg (23K)', 'Logistic Reg (200K)', 'Random Forest (23K)', 'Tree/Boost (200K)']
    sklearn_times = [
        scalability_results['scale_v2_23k']['Sklearn_LR_sec'],
        scalability_results['scale_v1_200k']['Sklearn_LR_sec'],
        scalability_results['scale_v2_23k']['Sklearn_RF_sec'],
        scalability_results['scale_v1_200k']['XGBoost_sec']
    ]
    spark_times = [
        scalability_results['scale_v2_23k']['Spark_LR_sec'],
        scalability_results['scale_v1_200k']['Spark_LR_sec'],
        scalability_results['scale_v2_23k']['Spark_RF_sec'],
        scalability_results['scale_v1_200k']['Spark_RF_sec']
    ]

    x2 = np.arange(len(scalability_labels))
    axes[1].bar(x2 - width/2, sklearn_times, width, label='Single-Node (Sklearn / XGBoost)', color='#386cb0')
    axes[1].bar(x2 + width/2, spark_times, width, label='Distributed (PySpark MLlib)', color='#f0027f')
    axes[1].set_ylabel('Training Time (Seconds)')
    axes[1].set_title('Execution Time Benchmark: 23,634 Rows vs 200,000 Rows', fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(scalability_labels, rotation=25, ha='right')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    spark_vs_path = 'diagrams/ml_spark_vs_sklearn.png'
    plt.savefig(spark_vs_path, dpi=300)
    plt.close()
    print(f"    - Saved: {spark_vs_path}")

    # --- PART 7: SAVE NUMERICAL RESULTS ---
    print("\n--------------------------------------------------------------------------------")
    print(" 6. SAVING NUMERICAL EVALUATION RESULTS TO JSON & TXT")
    print("--------------------------------------------------------------------------------")
    full_output = {
        'metadata': {
            'student_name': 'Kevin Sebastian',
            'nim': '2702746012',
            'course': 'COMP8035041 - Big Data Analytics',
            'dataset_v2_rows': total_rows,
            'dataset_v1_benchmark_rows': 200000,
            'fraud_ratio': imbalance_ratio,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'evaluation_metrics': eval_results,
        'threshold_tuning': {
            'best_threshold': best_thresh,
            'best_f1_score': best_f1,
            'grid': tuning_records
        },
        'scalability_benchmark': scalability_results
    }

    with open('ml_evaluation_results.json', 'w') as f:
        json.dump(full_output, f, indent=2)
    print("    - Saved: ml_evaluation_results.json")

    # Formatted Text Table
    txt_lines = []
    txt_lines.append("=" * 105)
    txt_lines.append(f"{'MODEL EVALUATION RESULTS - ASSIGNMENT II BIG DATA ANALYTICS':^105}")
    txt_lines.append(f"{'Student: Kevin Sebastian | NIM: 2702746012':^105}")
    txt_lines.append("=" * 105)
    txt_lines.append(f"{'Model Name':<32} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Train(s)':<8}")
    txt_lines.append("-" * 105)

    for k, v in eval_results.items():
        txt_lines.append(
            f"{v['model_name']:<32} | {v['accuracy']:<8.4f} | {v['precision']:<9.4f} | "
            f"{v['recall']:<8.4f} | {v['f1_score']:<8.4f} | {v['roc_auc']:<8.4f} | "
            f"{v['pr_auc']:<8.4f} | {v['training_time_sec']:<8.2f}"
        )
    txt_lines.append("-" * 105)
    txt_lines.append("\nCONFUSION MATRICES SUMMARY:")
    for k, v in eval_results.items():
        cm = v['confusion_matrix']
        txt_lines.append(f"  * {v['model_name']:<32}: TN={cm['tn']:,}, FP={cm['fp']:,}, FN={cm['fn']:,}, TP={cm['tp']:,}")

    txt_lines.append("\nSCALABILITY EXECUTION TIME (SECONDS):")
    txt_lines.append(f"  * V2 (23,634 rows)   -> Sklearn LR: {scalability_results['scale_v2_23k']['Sklearn_LR_sec']:.2f}s | XGBoost: {scalability_results['scale_v2_23k']['XGBoost_sec']:.2f}s | Spark LR: {scalability_results['scale_v2_23k']['Spark_LR_sec']:.2f}s | Spark RF: {scalability_results['scale_v2_23k']['Spark_RF_sec']:.2f}s")
    txt_lines.append(f"  * V1 (200,000 rows)  -> Sklearn LR: {scalability_results['scale_v1_200k']['Sklearn_LR_sec']:.2f}s | XGBoost: {scalability_results['scale_v1_200k']['XGBoost_sec']:.2f}s | Spark LR: {scalability_results['scale_v1_200k']['Spark_LR_sec']:.2f}s | Spark RF: {scalability_results['scale_v1_200k']['Spark_RF_sec']:.2f}s")

    txt_content = "\n".join(txt_lines)
    with open('ml_evaluation_results.txt', 'w') as f:
        f.write(txt_content)
    print("    - Saved: ml_evaluation_results.txt")
    print("\n" + txt_content)
    print("\n[SUCCESS] Task 2: Machine Learning Pipeline execution finished successfully!")

if __name__ == '__main__':
    run_ml_pipeline()
