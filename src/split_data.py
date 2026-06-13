import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
TRAIN_DIR = os.path.join(PROCESSED_DIR, 'train')
TEST_DIR = os.path.join(PROCESSED_DIR, 'test')

RANDOM_STATE = 42
TEST_SIZE = 0.2

print("=" * 60)
print("CHIA DỮ LIỆU TRAIN/TEST (80/20)")
print("=" * 60)

# ============================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================
ratings = pd.read_csv(os.path.join(RAW_DIR, 'ratings.csv'))
movies = pd.read_csv(os.path.join(RAW_DIR, 'movies.csv'))
transactions = pd.read_csv(os.path.join(PROCESSED_DIR, 'transactions.csv'))

print(f"\nRatings: {ratings.shape}")
print(f"Transactions: {transactions.shape}")
print(f"Số user trong ratings: {ratings['userId'].nunique()}")
print(f"Số user trong transactions: {transactions['userId'].nunique()}")

# ============================================================
# 2. CHIA USER ID
# ============================================================
all_users = ratings['userId'].unique()
train_users, test_users = train_test_split(
    all_users,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"\nTổng số user: {len(all_users)}")
print(f"Train users: {len(train_users)} ({len(train_users)/len(all_users)*100:.0f}%)")
print(f"Test users:  {len(test_users)} ({len(test_users)/len(all_users)*100:.0f}%)")

# ============================================================
# 3. CHIA RATINGS
# ============================================================
print("\n" + "-" * 40)
print("CHIA RATINGS")
print("-" * 40)

ratings_train = ratings[ratings['userId'].isin(train_users)].copy()
ratings_test = ratings[ratings['userId'].isin(test_users)].copy()

print(f"  Ratings train: {ratings_train.shape} ({len(ratings_train)/len(ratings)*100:.1f}%)")
print(f"  Ratings test:  {ratings_test.shape} ({len(ratings_test)/len(ratings)*100:.1f}%)")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

ratings_train.to_csv(os.path.join(TRAIN_DIR, 'ratings.csv'), index=False)
ratings_test.to_csv(os.path.join(TEST_DIR, 'ratings.csv'), index=False)
print(f"  -> {TRAIN_DIR}\\ratings.csv")
print(f"  -> {TEST_DIR}\\ratings.csv")

# ============================================================
# 4. CHIA TRANSACTIONS
# ============================================================
print("\n" + "-" * 40)
print("CHIA TRANSACTIONS")
print("-" * 40)

transactions_train = transactions[transactions['userId'].isin(train_users)].copy()
transactions_test = transactions[transactions['userId'].isin(test_users)].copy()

print(f"  Transactions train: {transactions_train.shape} ({len(transactions_train)/len(transactions)*100:.1f}%)")
print(f"  Transactions test:  {transactions_test.shape} ({len(transactions_test)/len(transactions)*100:.1f}%)")

transactions_train.to_csv(os.path.join(TRAIN_DIR, 'transactions.csv'), index=False)
transactions_test.to_csv(os.path.join(TEST_DIR, 'transactions.csv'), index=False)
print(f"  -> {TRAIN_DIR}\\transactions.csv")
print(f"  -> {TEST_DIR}\\transactions.csv")

# ============================================================
# 5. THỐNG KÊ
# ============================================================
print("\n" + "-" * 40)
print("THỐNG KÊ CHI TIẾT")
print("-" * 40)

print(f"\nRatings:")
print(f"  Train: {ratings_train['userId'].nunique()} users, {ratings_train['movieId'].nunique()} movies, {len(ratings_train)} ratings")
print(f"  Test:  {ratings_test['userId'].nunique()} users, {ratings_test['movieId'].nunique()} movies, {len(ratings_test)} ratings")

print(f"\nTransactions:")
print(f"  Train: {len(transactions_train)} giao dịch")
print(f"  Test:  {len(transactions_test)} giao dịch")

print("\n" + "=" * 60)
print("HOÀN TẤT!")
print("=" * 60)
