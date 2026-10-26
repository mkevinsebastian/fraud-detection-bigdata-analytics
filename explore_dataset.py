"""
Eksplorasi Dataset: Fraudulent E-Commerce Transactions
Assignment I - Big Data Analytics (COMP8035041)
"""

import pandas as pd

DATA_DIR = r"D:\Kevin\Binus S2\Semester 3\Big Data Analytics\Fraud_Detection_Project\dataset"
FILE_V1 = f"{DATA_DIR}\\Fraudulent_E-Commerce_Transaction_Data.csv"
FILE_V2 = f"{DATA_DIR}\\Fraudulent_E-Commerce_Transaction_Data_2.csv"

OUTPUT_FILE = r"D:\Kevin\Binus S2\Semester 3\Big Data Analytics\Fraud_Detection_Project\dataset_exploration_results.txt"


def analyze_dataset(path, label, output_lines):
    output_lines.append("\n" + "=" * 70)
    output_lines.append(f"DATASET: {label}")
    output_lines.append(f"File: {path}")
    output_lines.append("=" * 70)

    df = pd.read_csv(path)

    output_lines.append("\n--- 1. INFO DASAR ---")
    output_lines.append(f"Jumlah baris: {len(df):,}")
    output_lines.append(f"Jumlah kolom: {len(df.columns)}")
    output_lines.append(f"Kolom: {list(df.columns)}")

    output_lines.append("\n--- 2. MISSING VALUES ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(3)
    any_missing = False
    for col in df.columns:
        if missing[col] > 0:
            any_missing = True
            output_lines.append(f"  {col}: {missing[col]:,} ({missing_pct[col]}%)")
    if not any_missing:
        output_lines.append("  Tidak ada missing values.")

    output_lines.append("\n--- 3. DUPLICATES ---")
    dup_count = df.duplicated().sum()
    dup_tx_id = df.duplicated(subset=['Transaction ID']).sum() if 'Transaction ID' in df.columns else 'N/A'
    output_lines.append(f"  Baris duplikat (semua kolom): {dup_count:,}")
    output_lines.append(f"  Transaction ID duplikat: {dup_tx_id}")

    if 'Is Fraudulent' in df.columns:
        output_lines.append("\n--- 4. DISTRIBUSI FRAUD (Is Fraudulent) ---")
        fraud_counts = df['Is Fraudulent'].value_counts().sort_index()
        fraud_pct = (df['Is Fraudulent'].value_counts(normalize=True).sort_index() * 100).round(3)
        for idx in fraud_counts.index:
            label_name = "Legitimate (0)" if idx == 0 else "Fraud (1)"
            output_lines.append(f"  {label_name}: {fraud_counts[idx]:,} ({fraud_pct[idx]}%)")
        if 1 in fraud_counts and 0 in fraud_counts:
            ratio = fraud_counts[0] / fraud_counts[1]
            output_lines.append(f"  Class Imbalance Ratio: {ratio:.1f}:1 (Legitimate:Fraud)")

    output_lines.append("\n--- 5. STATISTIK FITUR NUMERIKAL ---")
    numerical_cols = ['Transaction Amount', 'Customer Age', 'Quantity', 'Account Age Days', 'Transaction Hour']
    numerical_cols = [c for c in numerical_cols if c in df.columns]
    stats = df[numerical_cols].describe().round(2)
    output_lines.append(stats.to_string())

    output_lines.append("\n--- 6. DISTRIBUSI FITUR KATEGORIKAL ---")
    categorical_cols = ['Payment Method', 'Product Category', 'Device Used']
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    for col in categorical_cols:
        output_lines.append(f"\n  {col}:")
        counts = df[col].value_counts()
        pcts = (df[col].value_counts(normalize=True) * 100).round(2)
        for val in counts.index:
            output_lines.append(f"    {val}: {counts[val]:,} ({pcts[val]}%)")

    if 'Is Fraudulent' in df.columns:
        output_lines.append("\n--- 7. FRAUD RATE PER KATEGORI ---")
        for col in categorical_cols:
            output_lines.append(f"\n  Fraud Rate by {col}:")
            rate = (df.groupby(col)['Is Fraudulent'].mean() * 100).round(2).sort_values(ascending=False)
            for val, r in rate.items():
                output_lines.append(f"    {val}: {r}%")

        if 'Transaction Hour' in df.columns:
            output_lines.append("\n  Fraud Rate by Transaction Hour:")
            rate_hour = (df.groupby('Transaction Hour')['Is Fraudulent'].mean() * 100).round(2)
            for h, r in rate_hour.items():
                output_lines.append(f"    Hour {h:02d}: {r}%")

        if 'Account Age Days' in df.columns:
            output_lines.append("\n  Fraud Rate by Account Age Bucket:")
            df['_age_bucket'] = pd.cut(df['Account Age Days'], bins=[-1, 30, 90, 365, 100000],
                                        labels=['0-30 days', '31-90 days', '91-365 days', '>365 days'])
            rate_age = (df.groupby('_age_bucket')['Is Fraudulent'].mean() * 100).round(2)
            for bucket, r in rate_age.items():
                output_lines.append(f"    {bucket}: {r}%")

        if 'Transaction Amount' in df.columns:
            output_lines.append("\n  Fraud Rate by Transaction Amount Bucket:")
            df['_amt_bucket'] = pd.cut(df['Transaction Amount'], bins=[0, 100, 500, 1000, 1000000],
                                        labels=['$0-100', '$100-500', '$500-1000', '>$1000'])
            rate_amt = (df.groupby('_amt_bucket')['Is Fraudulent'].mean() * 100).round(2)
            for bucket, r in rate_amt.items():
                output_lines.append(f"    {bucket}: {r}%")

    output_lines.append("\n--- 8. ANALISIS SHIPPING VS BILLING ADDRESS ---")
    if 'Shipping Address' in df.columns and 'Billing Address' in df.columns:
        mismatch = (df['Shipping Address'] != df['Billing Address']).sum()
        mismatch_pct = round(mismatch / len(df) * 100, 2)
        output_lines.append(f"  Address mismatch (shipping != billing): {mismatch:,} ({mismatch_pct}%)")
        if 'Is Fraudulent' in df.columns:
            df['_addr_mismatch'] = (df['Shipping Address'] != df['Billing Address']).astype(int)
            fraud_rate_mismatch = (df.groupby('_addr_mismatch')['Is Fraudulent'].mean() * 100).round(2)
            output_lines.append(f"  Fraud rate saat address match: {fraud_rate_mismatch.get(0, 'N/A')}%")
            output_lines.append(f"  Fraud rate saat address mismatch: {fraud_rate_mismatch.get(1, 'N/A')}%")

    return df


def main():
    output_lines = []
    output_lines.append("HASIL EKSPLORASI DATASET - Fraudulent E-Commerce Transactions")
    output_lines.append("Generated for Assignment I - Big Data Analytics (COMP8035041)")

    try:
        analyze_dataset(FILE_V2, "Version 2 (Smaller/Testing Dataset)", output_lines)
    except Exception as e:
        output_lines.append(f"\nError analyzing V2: {e}")

    try:
        analyze_dataset(FILE_V1, "Version 1 (Full Dataset)", output_lines)
    except Exception as e:
        output_lines.append(f"\nError analyzing V1: {e}")

    result_text = "\n".join(output_lines)
    print(result_text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(f"\n\nHasil lengkap disimpan ke: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
