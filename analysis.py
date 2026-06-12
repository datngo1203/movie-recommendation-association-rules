import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. ĐỌC & THU THẬP DỮ LIỆU
# ============================================================
print("=" * 60)
print("1. ĐỌC & THU THẬP DỮ LIỆU")
print("=" * 60)

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')
links = pd.read_csv('links.csv')

print(f"Movies: {movies.shape}")
print(f"Ratings: {ratings.shape}")
print(f"Tags: {tags.shape}")
print(f"Links: {links.shape}")

# ============================================================
# 2. TIỀN XỬ LÝ DỮ LIỆU
# ============================================================
print("\n" + "=" * 60)
print("2. TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 60)

ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
movies['genres_list'] = movies['genres'].str.split('|')

all_genres = set()
for glist in movies['genres_list']:
    all_genres.update(glist)
for g in all_genres:
    movies[g] = movies['genres_list'].apply(lambda x: 1 if g in x else 0)

print(f"Tổng số phim: {len(movies)}")
print(f"Tổng số user: {ratings['userId'].nunique()}")
print(f"Tổng số rating: {len(ratings)}")
print(f"Tổng số genres: {len(all_genres)}")

# ============================================================
# 3. PHÂN TÍCH
# ============================================================
print("\n" + "=" * 60)
print("3. PHÂN TÍCH DỮ LIỆU")
print("=" * 60)

# Thống kê ratings
print("\n--- Thống kê ratings ---")
print(ratings['rating'].describe())

# Top phim nhiều rating nhất
top_rated = ratings['movieId'].value_counts().head(10).reset_index()
top_rated.columns = ['movieId', 'num_ratings']
top_rated = top_rated.merge(movies[['movieId', 'title']], on='movieId')
print("\n--- Top 10 phim nhiều rating nhất ---")
print(top_rated[['title', 'num_ratings']])

# Thống kê users
user_stats = ratings.groupby('userId')['rating'].agg(['count', 'mean'])
print(f"\nSố lượng user: {len(user_stats)}")
print(f"Trung bình số rating/user: {user_stats['count'].mean():.1f}")

# Số lượng phim theo năm
movies_per_year = movies['year'].value_counts().sort_index()

# ============================================================
# 4. TRỰC QUAN HÓA - 4 Biểu đồ riêng biệt
# ============================================================
print("\n" + "=" * 60)
print("4. TRỰC QUAN HÓA DỮ LIỆU")
print("=" * 60)

os.makedirs('charts', exist_ok=True)

# Chart 1: Top phim được xem nhiều nhất
top15 = ratings['movieId'].value_counts().head(15).reset_index()
top15.columns = ['movieId', 'count']
top15 = top15.merge(movies[['movieId', 'title']], on='movieId')

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(range(len(top15)), top15['count'].values, color='coral', edgecolor='darkred', linewidth=0.5)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels([t[:50] for t in top15['title']], fontsize=10)
ax.set_xlabel('Số lượng rating', fontsize=12)
ax.set_title('Top 15 phim được xem nhiều nhất', fontsize=16, fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(top15['count']):
    ax.text(v + 2, i, str(v), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('charts/top_movies.png', dpi=150)
plt.close()
print("  -> charts/top_movies.png")

# Chart 2: Phân bố rating
rating_counts = ratings['rating'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(rating_counts)))
bars = ax.bar(rating_counts.index.astype(str), rating_counts.values, color=colors, edgecolor='gray', linewidth=0.8)
ax.set_xlabel('Rating', fontsize=12)
ax.set_ylabel('Số lượng', fontsize=12)
ax.set_title('Phân bố điểm rating', fontsize=16, fontweight='bold')
for bar, v in zip(bars, rating_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, str(v),
            ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('charts/rating_distribution.png', dpi=150)
plt.close()
print("  -> charts/rating_distribution.png")

# Chart 3: Số lượng người dùng (phân phối số rating mỗi user)
user_rating_counts = ratings.groupby('userId').size()
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(user_rating_counts, bins=60, color='steelblue', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Số lượng rating', fontsize=12)
ax.set_ylabel('Số lượng user', fontsize=12)
ax.set_title('Phân phối số lượng rating theo người dùng', fontsize=14, fontweight='bold')
ax.axvline(user_rating_counts.mean(), color='red', linestyle='--', linewidth=1.5,
           label=f'Trung bình: {user_rating_counts.mean():.1f}')
ax.axvline(user_rating_counts.median(), color='orange', linestyle=':', linewidth=1.5,
           label=f'Trung vị: {user_rating_counts.median():.1f}')
ax.legend(fontsize=10)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('charts/user_distribution.png', dpi=150)
plt.close()
print("  -> charts/user_distribution.png")

# Chart 4: Số lượng phim
fig, ax = plt.subplots(figsize=(14, 6))

# Biểu đồ con 1: Số phim theo năm
ax1 = fig.add_subplot(121)
years = movies['year'].dropna().astype(int)
year_counts = years.value_counts().sort_index()
ax1.bar(year_counts.index, year_counts.values, color='mediumseagreen', edgecolor='white', linewidth=0.3, width=0.8)
ax1.set_xlabel('Năm sản xuất', fontsize=11)
ax1.set_ylabel('Số lượng phim', fontsize=11)
ax1.set_title('Số lượng phim theo năm', fontsize=14, fontweight='bold')

# Biểu đồ con 2: Số phim theo genre
ax2 = fig.add_subplot(122)
genre_counts = movies[list(all_genres)].sum().sort_values(ascending=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(genre_counts)))
ax2.barh(range(len(genre_counts)), genre_counts.values, color=colors, edgecolor='gray', linewidth=0.5)
ax2.set_yticks(range(len(genre_counts)))
ax2.set_yticklabels(genre_counts.index, fontsize=9)
ax2.set_xlabel('Số lượng phim', fontsize=11)
ax2.set_title('Số lượng phim theo thể loại', fontsize=14, fontweight='bold')
for i, v in enumerate(genre_counts.values):
    ax2.text(v + 10, i, str(v), va='center', fontsize=8)

plt.tight_layout()
plt.savefig('charts/movie_statistics.png', dpi=150)
plt.close()
print("  -> charts/movie_statistics.png")

# ============================================================
# 5. BÁO CÁO TỔNG QUAN
# ============================================================
print("\n" + "=" * 60)
print("5. BÁO CÁO TỔNG QUAN")
print("=" * 60)
print(f"  - Tổng số phim:            {len(movies)}")
print(f"  - Tổng số người dùng:      {ratings['userId'].nunique()}")
print(f"  - Tổng số rating:          {len(ratings):,}")
print(f"  - Tổng số tag:             {len(tags):,}")
print(f"  - Số thể loại:             {len(all_genres)}")
print(f"  - Rating trung bình:       {ratings['rating'].mean():.2f}")
print(f"  - Khoảng thời gian:        {ratings['datetime'].min().year} - {ratings['datetime'].max().year}")
print(f"  - User trung bình rating:  {user_rating_counts.mean():.1f}")
print(f"  - User tích cực nhất:      {user_rating_counts.max()} ratings")
print("\n" + "=" * 60)
print("HOÀN TẤT! Biểu đồ được lưu trong thư mục charts/")
print("=" * 60)
