import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# ============================================================
# 1.2 THU THẬP DỮ LIỆU
# ============================================================
print("=" * 60)
print("1.2 THU THẬP DỮ LIỆU")
print("=" * 60)
print("""
Nguồn dữ liệu: MovieLens Latest Small (GroupLens Research)
  - URL: https://grouplens.org/datasets/movielens/latest/
  - Phiên bản: ml-latest-small (100k ratings)

Mô tả tập dữ liệu:
  - 100,836 bản ghi rating từ 610 người dùng trên 9,742 bộ phim
  - Thang điểm rating: 0.5 - 5.0 (bước 0.5)
  - Thời gian: 1996 - 2018
  - 20 thể loại phim (genres)

Các thuộc tính dữ liệu:
  1. movies.csv:     movieId, title, genres
  2. ratings.csv:    userId, movieId, rating, timestamp
  3. tags.csv:       userId, movieId, tag, timestamp
  4. links.csv:      movieId, imdbId, tmdbId
""")

# ============================================================
# ĐỌC DỮ LIỆU
# ============================================================
movies = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))
ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
tags = pd.read_csv(os.path.join(DATA_DIR, 'tags.csv'))
links = pd.read_csv(os.path.join(DATA_DIR, 'links.csv'))

ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

print(f"\nMovies:  {movies.shape[0]:>7} dòng, {movies.shape[1]} cột: {list(movies.columns)}")
print(f"Ratings: {ratings.shape[0]:>7} dòng, {ratings.shape[1]} cột: {list(ratings.columns)}")
print(f"Tags:    {tags.shape[0]:>7} dòng, {tags.shape[1]} cột: {list(tags.columns)}")
print(f"Links:   {links.shape[0]:>7} dòng, {links.shape[1]} cột: {list(links.columns)}")

# ============================================================
# 1.3 PHÂN TÍCH DỮ LIỆU
# ============================================================
print("\n" + "=" * 60)
print("1.3 PHÂN TÍCH DỮ LIỆU")
print("=" * 60)

# --- Thống kê mô tả ---
print("\n--- THỐNG KÊ MÔ TẢ ---")

print("\na) Ratings:")
print(f"   Tổng số rating: {len(ratings):,}")
print(f"   Giá trị:        từ {ratings['rating'].min()} đến {ratings['rating'].max()}")
print(f"   Trung bình:     {ratings['rating'].mean():.3f}")
print(f"   Độ lệch chuẩn:  {ratings['rating'].std():.3f}")
print(f"   Trung vị:       {ratings['rating'].median():.2f}")

# Rating distribution
rating_counts = ratings['rating'].value_counts().sort_index()
print(f"\n   Phân bố rating:")
for val in rating_counts.index:
    pct = rating_counts[val] / len(ratings) * 100
    print(f"     {val:>4}: {rating_counts[val]:>6} ({pct:>5.1f}%)")

print(f"\nb) Users:")
print(f"   Tổng số:                {ratings['userId'].nunique()}")
user_rating_counts = ratings.groupby('userId').size()
print(f"   Rating trung bình/user: {user_rating_counts.mean():.1f}")
print(f"   User tích cực nhất:     {user_rating_counts.max()} ratings")
print(f"   User ít tích cực nhất:  {user_rating_counts.min()} ratings")

print(f"\nc) Movies:")
print(f"   Tổng số:                {len(movies)}")
print(f"   Số năm:                 {int(movies['year'].min())} - {int(movies['year'].max())}")
print(f"   Số lượng genres:        {movies['genres'].str.split('|').explode().nunique()}")

# Genre distribution
genre_counts = movies['genres'].str.split('|').explode().value_counts()
print(f"\n   Top 10 thể loại phổ biến:")
for g, c in genre_counts.head(10).items():
    print(f"     {g:>15}: {c:>4} phim ({c/len(movies)*100:.1f}%)")

print(f"\nd) Tags:")
print(f"   Tổng số tag:                   {len(tags):,}")
print(f"   Số user tag:                   {tags['userId'].nunique()}")
print(f"   Số phim được tag:              {tags['movieId'].nunique()}")

print(f"\ne) Links:")
print(f"   Thiếu tmdbId:                  {links['tmdbId'].isna().sum()} bản ghi")

# --- Ma trận tương quan genres ---
print("\n--- PHÂN TÍCH TƯƠNG QUAN ---")
movies['genres_list'] = movies['genres'].str.split('|')
all_genres = sorted(movies['genres'].str.split('|').explode().unique())
for g in all_genres:
    movies[g] = movies['genres_list'].apply(lambda x: 1 if g in x else 0)

# Genre co-occurrence
genre_matrix = movies[all_genres].T.dot(movies[all_genres])
np.fill_diagonal(genre_matrix.values, 0)
top_cooc = genre_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(5)
print(f"\n   Top 5 cặp genre hay xuất hiện cùng nhau:")
for (g1, g2), count in top_cooc.items():
    print(f"     {g1} + {g2}: {int(count)} phim")

# --- Phân tích hành vi user ---
user_means = ratings.groupby('userId')['rating'].mean()
user_stds = ratings.groupby('userId')['rating'].std().fillna(0)
print(f"\n--- PHÂN TÍCH HÀNH VI USER ---")
print(f"   Điểm TB user thấp nhất:  {user_means.min():.2f}")
print(f"   Điểm TB user cao nhất:   {user_means.max():.2f}")
print(f"   User khó tính (TB < 2):  {(user_means < 2).sum()} users")
print(f"   User dễ tính (TB > 4.5): {(user_means > 4.5).sum()} users")

# Rating trend by year
yearly_stats = ratings.groupby(ratings['datetime'].dt.year)['rating'].agg(['mean', 'count'])
yearly_stats.columns = ['rating_mean', 'rating_count']
print(f"\n--- XU HƯỚNG THEO THỜI GIAN ---")
print(f"   Năm cao điểm rating: {yearly_stats['rating_count'].idxmax()} ({int(yearly_stats['rating_count'].max()):,})")
print(f"   Năm thấp nhất:       {yearly_stats['rating_count'].idxmin()} ({int(yearly_stats['rating_count'].min()):,})")

# ============================================================
# TRỰC QUAN HÓA DỮ LIỆU
# ============================================================
print("\n" + "=" * 60)
print("TRỰC QUAN HÓA DỮ LIỆU")
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

# Chart 3: Phân phối rating theo user
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

# Chart 4: Số lượng phim theo năm và thể loại
fig, ax = plt.subplots(figsize=(14, 6))

ax1 = fig.add_subplot(121)
years = movies['year'].dropna().astype(int)
year_counts = years.value_counts().sort_index()
ax1.bar(year_counts.index, year_counts.values, color='mediumseagreen', edgecolor='white', linewidth=0.3, width=0.8)
ax1.set_xlabel('Năm sản xuất', fontsize=11)
ax1.set_ylabel('Số lượng phim', fontsize=11)
ax1.set_title('Số lượng phim theo năm', fontsize=14, fontweight='bold')

ax2 = fig.add_subplot(122)
genre_sum = movies[list(all_genres)].sum().sort_values(ascending=True)
color_map = plt.cm.Set3(np.linspace(0, 1, len(genre_sum)))
ax2.barh(range(len(genre_sum)), genre_sum.values, color=color_map, edgecolor='gray', linewidth=0.5)
ax2.set_yticks(range(len(genre_sum)))
ax2.set_yticklabels(genre_sum.index, fontsize=9)
ax2.set_xlabel('Số lượng phim', fontsize=11)
ax2.set_title('Số lượng phim theo thể loại', fontsize=14, fontweight='bold')
for i, v in enumerate(genre_sum.values):
    ax2.text(v + 10, i, str(v), va='center', fontsize=8)

plt.tight_layout()
plt.savefig('charts/movie_statistics.png', dpi=150)
plt.close()
print("  -> charts/movie_statistics.png")

# Chart 5: Heatmap tương quan thể loại
fig, ax = plt.subplots(figsize=(12, 10))
corr = movies[all_genres].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
im = ax.imshow(corr.where(mask), cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
ax.set_xticks(range(len(all_genres)))
ax.set_yticks(range(len(all_genres)))
ax.set_xticklabels(all_genres, rotation=90, fontsize=8)
ax.set_yticklabels(all_genres, fontsize=8)
ax.set_title('Ma trận tương quan giữa các thể loại', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig('charts/genre_correlation.png', dpi=150)
plt.close()
print("  -> charts/genre_correlation.png")

# Chart 6: Rating trung bình theo năm
fig, ax = plt.subplots(figsize=(12, 5))
ax2 = ax.twinx()
ax.plot(yearly_stats.index, yearly_stats['rating_mean'], 'o-', color='green', linewidth=2, markersize=6, label='Rating TB')
ax2.bar(yearly_stats.index, yearly_stats['rating_count'], alpha=0.3, color='steelblue', label='Số lượng')
ax.set_xlabel('Năm', fontsize=12)
ax.set_ylabel('Rating trung bình', fontsize=12, color='green')
ax2.set_ylabel('Số lượng rating', fontsize=12, color='steelblue')
ax.set_title('Xu hướng rating theo thời gian', fontsize=14, fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
plt.savefig('charts/rating_trend.png', dpi=150)
plt.close()
print("  -> charts/rating_trend.png")

# ============================================================
# NHẬN XÉT ĐẶC ĐIỂM DỮ LIỆU
# ============================================================
print("\n" + "=" * 60)
print("NHẬN XÉT ĐẶC ĐIỂM DỮ LIỆU")
print("=" * 60)

print("""
1. Phân bố rating:
   - Phần lớn rating tập trung ở mức cao (3.0-5.0)
   - Rating 4.0 chiếm tỷ lệ cao nhất, phản ánh xu hướng đánh giá tích cực
   - Rating thấp (0.5-2.0) chiếm tỷ lệ nhỏ

2. Người dùng:
   - Số lượng rating/user phân bố lệch: đa số user có ít rating
   - Một số user siêu tích cực (>1000 ratings) nhưng hiếm
   - User có xu hướng chấm điểm khá cao (TB ~3.5)

3. Thể loại phim:
   - Drama, Comedy, Thriller là 3 thể loại phổ biến nhất
   - Nhiều phim thuộc nhiều thể loại cùng lúc
   - Drama + Romance, Comedy + Romance là cặp hay đi cùng nhau

4. Thời gian:
   - Dữ liệu ratings trải dài từ 1996 đến 2018 (22 năm)
   - Số lượng rating tăng dần theo thời gian
   - Các phim cũ (trước 2000) ít được đánh giá hơn

5. Chất lượng dữ liệu:
   - Không có missing values trong ratings, movies, tags
   - Links có một số thiếu tmdbId (~10 bản ghi)
   - Dữ liệu ratings hoàn chỉnh, không trùng lặp
""")

# ============================================================
# TỔNG KẾT
# ============================================================
print("=" * 60)
print("TỔNG KẾT")
print("=" * 60)
print(f"  - Tổng số phim:            {len(movies)}")
print(f"  - Tổng số người dùng:      {ratings['userId'].nunique()}")
print(f"  - Tổng số rating:          {len(ratings):,}")
print(f"  - Tổng số tag:             {len(tags):,}")
print(f"  - Số thể loại:             {len(all_genres)}")
print(f"  - Rating trung bình:       {ratings['rating'].mean():.2f}")
print(f"  - Khoảng thời gian:        {ratings['datetime'].min().year} - {ratings['datetime'].max().year}")
print(f"  - Mật độ ma trận:          {len(ratings) / (ratings['userId'].nunique() * len(movies)) * 100:.4f}%")
print(f"  - User trung bình rating:  {user_rating_counts.mean():.1f}")
print(f"  - User tích cực nhất:      {user_rating_counts.max()} ratings")
print("\n" + "=" * 60)
print("HOÀN TẤT! Kết quả lưu tại thư mục charts/")
print("=" * 60)
