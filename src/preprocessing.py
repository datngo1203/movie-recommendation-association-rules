import pandas as pd
import numpy as np
import os
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 60)
print("1.4 TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 60)

# ============================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================
movies = pd.read_csv(os.path.join(RAW_DIR, 'movies.csv'))
ratings = pd.read_csv(os.path.join(RAW_DIR, 'ratings.csv'))
tags = pd.read_csv(os.path.join(RAW_DIR, 'tags.csv'))
links = pd.read_csv(os.path.join(RAW_DIR, 'links.csv'))

print(f"\nMovies: {movies.shape}")
print(f"Ratings: {ratings.shape}")
print(f"Tags: {tags.shape}")
print(f"Links: {links.shape}")

# ============================================================
# 2. XỬ LÝ DỮ LIỆU THIẾU
# ============================================================
print("\n" + "-" * 40)
print("2. XỬ LÝ DỮ LIỆU THIẾU")
print("-" * 40)

print("\n--- Movies ---")
missing_movies = movies.isnull().sum()
print(missing_movies[missing_movies > 0] if missing_movies.any() else "  Không có dữ liệu thiếu")

print("\n--- Ratings ---")
missing_ratings = ratings.isnull().sum()
print(missing_ratings[missing_ratings > 0] if missing_ratings.any() else "  Không có dữ liệu thiếu")

print("\n--- Tags ---")
missing_tags = tags.isnull().sum()
print(missing_tags[missing_tags > 0] if missing_tags.any() else "  Không có dữ liệu thiếu")

print("\n--- Links ---")
missing_links = links.isnull().sum()
print(missing_links[missing_links > 0])
links_clean = links.dropna(subset=['tmdbId'])

# ============================================================
# 3. XỬ LÝ NGOẠI LAI
# ============================================================
print("\n" + "-" * 40)
print("3. XỬ LÝ NGOẠI LAI")
print("-" * 40)

# Ratings: kiểm tra rating ngoài khoảng [0.5, 5.0]
invalid_ratings = ratings[(ratings['rating'] < 0.5) | (ratings['rating'] > 5.0)]
print(f"  Rating ngoài [0.5, 5.0]: {len(invalid_ratings)}")
ratings = ratings[(ratings['rating'] >= 0.5) & (ratings['rating'] <= 5.0)]

# Phát hiện user bất thường (quá ít hoặc quá nhiều rating)
user_counts = ratings['userId'].value_counts()
Q1 = user_counts.quantile(0.25)
Q3 = user_counts.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_users = user_counts[(user_counts < lower_bound) | (user_counts > upper_bound)]
print(f"  User ngoại lai (IQR): {len(outlier_users)} users")
print(f"  - Ngưỡng dưới: {lower_bound:.1f}, Ngưỡng trên: {upper_bound:.1f}")

# ============================================================
# 4. CHUẨN HÓA DỮ LIỆU
# ============================================================
print("\n" + "-" * 40)
print("4. CHUẨN HÓA DỮ LIỆU")
print("-" * 40)

# Min-Max Scaling cho rating (về [0,1])
rating_min = ratings['rating'].min()
rating_max = ratings['rating'].max()
ratings['rating_norm'] = (ratings['rating'] - rating_min) / (rating_max - rating_min)
print(f"  Rating normalized: [{ratings['rating_norm'].min():.2f}, {ratings['rating_norm'].max():.2f}]")

# Chuẩn hóa timestamp về năm-tháng
ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['year'] = ratings['datetime'].dt.year
ratings['month'] = ratings['datetime'].dt.month
print(f"  Khoảng thời gian: {ratings['year'].min()} - {ratings['year'].max()}")

# ============================================================
# 5. MÃ HÓA DỮ LIỆU
# ============================================================
print("\n" + "-" * 40)
print("5. MÃ HÓA DỮ LIỆU")
print("-" * 40)

# Trích xuất năm từ title
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
movies.loc[movies['year'].isna(), 'year'] = movies['title'].str.extract(r'(\d{4})').astype(float)

# One-hot encoding cho genres
all_genres = sorted(movies['genres'].str.split('|').explode().unique())
for g in all_genres:
    movies[g] = movies['genres'].str.contains(re.escape(g), na=False).astype(int)

# Mã hóa genres_list dạng set
movies['genres_list'] = movies['genres'].str.split('|').apply(set)
movies['genre_count'] = movies['genres_list'].apply(len)

# Tạo rating_bin (nhóm rating)
ratings['rating_bin'] = pd.cut(ratings['rating'], bins=[0, 1, 2, 3, 4, 5],
                                labels=['Rất tệ', 'Tệ', 'Trung bình', 'Tốt', 'Rất tốt'],
                                include_lowest=True)

print(f"  Số genres: {len(all_genres)}")
print(f"  Movie năm: {int(movies['year'].min())} - {int(movies['year'].max())}")
print(f"  Rating bins: {ratings['rating_bin'].value_counts().to_dict()}")

# ============================================================
# 6. TĂNG CƯỜNG DỮ LIỆU
# ============================================================
print("\n" + "-" * 40)
print("6. TĂNG CƯỜNG DỮ LIỆU")
print("-" * 40)

# Rating statistics per user
user_stats = ratings.groupby('userId').agg(
    user_rating_count=('rating', 'count'),
    user_rating_mean=('rating', 'mean'),
    user_rating_std=('rating', 'std')
).fillna(0)
ratings = ratings.merge(user_stats, on='userId', how='left')

# Rating statistics per movie
movie_stats = ratings.groupby('movieId').agg(
    movie_rating_count=('rating', 'count'),
    movie_rating_mean=('rating', 'mean')
)
ratings = ratings.merge(movie_stats, on='movieId', how='left')

# Tính điểm weighted rating (IMDB-style)
C = ratings['rating'].mean()
m = ratings['movie_rating_count'].quantile(0.8)
ratings['weighted_rating'] = (ratings['movie_rating_count'] / (ratings['movie_rating_count'] + m)) * ratings['movie_rating_mean'] + \
                              (m / (ratings['movie_rating_count'] + m)) * C

print(f"  User features: user_rating_count, user_rating_mean, user_rating_std")
print(f"  Movie features: movie_rating_count, movie_rating_mean, weighted_rating")

# ============================================================
# 7. LƯU DỮ LIỆU ĐÃ XỬ LÝ
# ============================================================
print("\n" + "-" * 40)
print("7. LƯU DỮ LIỆU ĐÃ XỬ LÝ")
print("-" * 40)

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

movies_out = movies.drop(columns=['genres_list'])
movies_out.to_csv(os.path.join(PROCESSED_DIR, 'movies_processed.csv'), index=False)
ratings.to_csv(os.path.join(PROCESSED_DIR, 'ratings_processed.csv'), index=False)
tags.to_csv(os.path.join(PROCESSED_DIR, 'tags_processed.csv'), index=False)
links_clean.to_csv(os.path.join(PROCESSED_DIR, 'links_processed.csv'), index=False)

print(f"  movies_processed.csv: {movies_out.shape}")
print(f"  ratings_processed.csv: {ratings.shape}")
print(f"  tags_processed.csv: {tags.shape}")
print(f"  links_processed.csv: {links_clean.shape}")

# ============================================================
# 8. TẠO DỮ LIỆU CHUẨN CHO LUẬT KẾT HỢP
# ============================================================
print("  -> Tạo dữ liệu giao dịch cho luật kết hợp...")

# Tạo dữ liệu giao dịch: mỗi user là 1 giao dịch chứa danh sách phim đã xem (rating >= 3.5)
threshold = 3.5
transactions = ratings[ratings['rating'] >= threshold].groupby('userId')['movieId'].apply(list).reset_index()
transactions.columns = ['userId', 'movie_ids']

movies_lookup = movies[['movieId', 'title', 'genres_list']].copy()
transactions['movie_titles'] = transactions['movie_ids'].apply(
    lambda ids: [movies_lookup[movies_lookup['movieId'] == mid]['title'].values[0]
                 if len(movies_lookup[movies_lookup['movieId'] == mid]['title'].values) > 0 else str(mid)
                 for mid in ids]
)

transaction_df = transactions[['userId', 'movie_titles']].copy()
transaction_df.to_csv(os.path.join(PROCESSED_DIR, 'transactions.csv'), index=False)
print(f"  transactions.csv: {transaction_df.shape}")
print(f"  Tổng số giao dịch: {len(transaction_df)}")

# Tạo dữ liệu người dùng đã mã hóa
user_item_matrix = ratings.pivot_table(
    index='userId', columns='movieId', values='rating_norm'
).fillna(0)
print(f"  user_item_matrix: {user_item_matrix.shape}")

# Lưu danh sách genres
genre_list_df = pd.DataFrame({'genre': all_genres})
genre_list_df.to_csv(os.path.join(PROCESSED_DIR, 'genres.csv'), index=False)

print("\n" + "=" * 60)
print("HOÀN TẤT TIỀN XỬ LÝ!")
print("=" * 60)
