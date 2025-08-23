"""
協調フィルタリングレコメンドシステム
ユーザー間・アイテム間の類似性に基づく推薦エンジン
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFilteringRecommender:
    def __init__(self):
        """協調フィルタリング推薦システムの初期化"""
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.svd_model = None
        self.user_means = None
        self.global_mean = None
        
    def load_interaction_data(self, file_path):
        """ユーザーとアイテムの相互作用データを読み込み"""
        try:
            df = pd.read_csv(file_path)
            
            required_columns = ['user_id', 'product_id', 'rating']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"必要な列が不足: {missing_columns}")
                return None
            
            # 評価値の正規化（1-5スケール）
            if df['rating'].min() >= 0 and df['rating'].max() <= 1:
                df['rating'] = df['rating'] * 5  # 0-1 → 0-5
            
            print(f"相互作用データ読み込み完了:")
            print(f"  ユーザー数: {df['user_id'].nunique()}")
            print(f"  商品数: {df['product_id'].nunique()}")
            print(f"  評価数: {len(df)}")
            print(f"  スパース率: {1 - len(df) / (df['user_id'].nunique() * df['product_id'].nunique()):.3f}")
            
            return df
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return None
    
    def create_user_item_matrix(self, df):
        """ユーザー-アイテム行列の作成"""
        # ピボットテーブル作成
        user_item_matrix = df.pivot_table(
            index='user_id', 
            columns='product_id', 
            values='rating', 
            fill_value=0
        )
        
        self.user_item_matrix = user_item_matrix
        self.global_mean = df['rating'].mean()
        
        # ユーザー平均評価の計算
        user_ratings = df.groupby('user_id')['rating'].mean()
        self.user_means = user_ratings.reindex(user_item_matrix.index, fill_value=self.global_mean)
        
        print(f"ユーザー-アイテム行列作成完了: {user_item_matrix.shape}")
        
        return user_item_matrix
    
    def calculate_user_similarity(self, method='cosine'):
        """ユーザー間類似度の計算"""
        if self.user_item_matrix is None:
            print("ユーザー-アイテム行列が作成されていません")
            return None
        
        print("ユーザー間類似度を計算中...")
        
        if method == 'cosine':
            # コサイン類似度
            # 0評価を除外するため、非ゼロ要素のみで計算
            matrix_nonzero = self.user_item_matrix.copy()
            matrix_nonzero[matrix_nonzero == 0] = np.nan
            
            user_similarity = np.zeros((len(self.user_item_matrix), len(self.user_item_matrix)))
            
            for i in range(len(self.user_item_matrix)):
                for j in range(i, len(self.user_item_matrix)):
                    user_i = matrix_nonzero.iloc[i].dropna()
                    user_j = matrix_nonzero.iloc[j].dropna()
                    
                    # 共通評価アイテム
                    common_items = user_i.index.intersection(user_j.index)
                    
                    if len(common_items) > 0:
                        ratings_i = user_i.loc[common_items]
                        ratings_j = user_j.loc[common_items]
                        
                        # コサイン類似度計算
                        dot_product = np.dot(ratings_i, ratings_j)
                        norm_i = np.linalg.norm(ratings_i)
                        norm_j = np.linalg.norm(ratings_j)
                        
                        if norm_i > 0 and norm_j > 0:
                            similarity = dot_product / (norm_i * norm_j)
                        else:
                            similarity = 0
                    else:
                        similarity = 0
                    
                    user_similarity[i, j] = similarity
                    user_similarity[j, i] = similarity
        
        self.user_similarity = pd.DataFrame(
            user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print("ユーザー間類似度計算完了")
        
        return self.user_similarity
    
    def calculate_item_similarity(self, method='cosine'):
        """アイテム間類似度の計算"""
        if self.user_item_matrix is None:
            print("ユーザー-アイテム行列が作成されていません")
            return None
        
        print("アイテム間類似度を計算中...")
        
        # アイテム×ユーザー行列（転置）
        item_user_matrix = self.user_item_matrix.T
        
        if method == 'cosine':
            # 非ゼロ要素のみで類似度計算
            matrix_nonzero = item_user_matrix.copy()
            matrix_nonzero[matrix_nonzero == 0] = np.nan
            
            item_similarity = np.zeros((len(item_user_matrix), len(item_user_matrix)))
            
            for i in range(len(item_user_matrix)):
                for j in range(i, len(item_user_matrix)):
                    item_i = matrix_nonzero.iloc[i].dropna()
                    item_j = matrix_nonzero.iloc[j].dropna()
                    
                    # 共通評価ユーザー
                    common_users = item_i.index.intersection(item_j.index)
                    
                    if len(common_users) > 0:
                        ratings_i = item_i.loc[common_users]
                        ratings_j = item_j.loc[common_users]
                        
                        # コサイン類似度計算
                        dot_product = np.dot(ratings_i, ratings_j)
                        norm_i = np.linalg.norm(ratings_i)
                        norm_j = np.linalg.norm(ratings_j)
                        
                        if norm_i > 0 and norm_j > 0:
                            similarity = dot_product / (norm_i * norm_j)
                        else:
                            similarity = 0
                    else:
                        similarity = 0
                    
                    item_similarity[i, j] = similarity
                    item_similarity[j, i] = similarity
        
        self.item_similarity = pd.DataFrame(
            item_similarity,
            index=item_user_matrix.index,
            columns=item_user_matrix.index
        )
        
        print("アイテム間類似度計算完了")
        
        return self.item_similarity
    
    def matrix_factorization(self, n_components=50):
        """行列分解による次元削減"""
        if self.user_item_matrix is None:
            print("ユーザー-アイテム行列が作成されていません")
            return None
        
        print(f"行列分解を実行中（成分数: {n_components}）...")
        
        # SVD（特異値分解）
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        
        # 行列分解実行
        user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        item_factors = self.svd_model.components_
        
        print(f"行列分解完了")
        print(f"  説明可能分散比: {self.svd_model.explained_variance_ratio_.sum():.3f}")
        
        return user_factors, item_factors
    
    def predict_user_based(self, user_id, product_id, k=10):
        """ユーザーベース協調フィルタリング予測"""
        if self.user_similarity is None:
            print("ユーザー類似度が計算されていません")
            return self.global_mean
        
        if user_id not in self.user_item_matrix.index:
            return self.global_mean
        
        if product_id not in self.user_item_matrix.columns:
            return self.global_mean
        
        # 対象ユーザーの類似ユーザーを取得
        user_similarities = self.user_similarity.loc[user_id].sort_values(ascending=False)
        
        # 対象商品を評価済みの類似ユーザー
        similar_users = []
        for similar_user_id, similarity in user_similarities.items():
            if similar_user_id != user_id and similarity > 0:
                if self.user_item_matrix.loc[similar_user_id, product_id] > 0:
                    similar_users.append((similar_user_id, similarity))
                
                if len(similar_users) >= k:
                    break
        
        if len(similar_users) == 0:
            return self.user_means.loc[user_id]
        
        # 予測評価値計算
        numerator = 0
        denominator = 0
        
        user_mean = self.user_means.loc[user_id]
        
        for similar_user_id, similarity in similar_users:
            similar_user_rating = self.user_item_matrix.loc[similar_user_id, product_id]
            similar_user_mean = self.user_means.loc[similar_user_id]
            
            numerator += similarity * (similar_user_rating - similar_user_mean)
            denominator += abs(similarity)
        
        if denominator > 0:
            predicted_rating = user_mean + numerator / denominator
        else:
            predicted_rating = user_mean
        
        # 評価値を1-5の範囲にクリップ
        return np.clip(predicted_rating, 1, 5)
    
    def predict_item_based(self, user_id, product_id, k=10):
        """アイテムベース協調フィルタリング予測"""
        if self.item_similarity is None:
            print("アイテム類似度が計算されていません")
            return self.global_mean
        
        if user_id not in self.user_item_matrix.index:
            return self.global_mean
        
        if product_id not in self.user_item_matrix.columns:
            return self.global_mean
        
        # 対象商品の類似商品を取得
        item_similarities = self.item_similarity.loc[product_id].sort_values(ascending=False)
        
        # ユーザーが評価済みの類似商品
        similar_items = []
        for similar_item_id, similarity in item_similarities.items():
            if similar_item_id != product_id and similarity > 0:
                if self.user_item_matrix.loc[user_id, similar_item_id] > 0:
                    similar_items.append((similar_item_id, similarity))
                
                if len(similar_items) >= k:
                    break
        
        if len(similar_items) == 0:
            return self.global_mean
        
        # 予測評価値計算
        numerator = 0
        denominator = 0
        
        for similar_item_id, similarity in similar_items:
            user_rating = self.user_item_matrix.loc[user_id, similar_item_id]
            
            numerator += similarity * user_rating
            denominator += abs(similarity)
        
        if denominator > 0:
            predicted_rating = numerator / denominator
        else:
            predicted_rating = self.global_mean
        
        # 評価値を1-5の範囲にクリップ
        return np.clip(predicted_rating, 1, 5)
    
    def predict_matrix_factorization(self, user_id, product_id):
        """行列分解による予測"""
        if self.svd_model is None:
            print("行列分解モデルが訓練されていません")
            return self.global_mean
        
        if user_id not in self.user_item_matrix.index:
            return self.global_mean
        
        if product_id not in self.user_item_matrix.columns:
            return self.global_mean
        
        # ユーザーとアイテムのインデックス
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(product_id)
        
        # SVDモデルから予測
        user_factors = self.svd_model.transform(self.user_item_matrix)
        item_factors = self.svd_model.components_
        
        predicted_rating = np.dot(user_factors[user_idx], item_factors[:, item_idx])
        
        # 評価値を1-5の範囲にクリップ
        return np.clip(predicted_rating, 1, 5)
    
    def recommend_items(self, user_id, n_recommendations=10, method='item_based'):
        """アイテム推薦"""
        if user_id not in self.user_item_matrix.index:
            print(f"ユーザー {user_id} が見つかりません")
            return []
        
        # ユーザーが未評価のアイテムを取得
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index.tolist()
        
        if len(unrated_items) == 0:
            print(f"ユーザー {user_id} の未評価アイテムがありません")
            return []
        
        # 各未評価アイテムの予測評価値を計算
        predictions = []
        
        for item_id in unrated_items:
            if method == 'user_based':
                predicted_rating = self.predict_user_based(user_id, item_id)
            elif method == 'item_based':
                predicted_rating = self.predict_item_based(user_id, item_id)
            elif method == 'matrix_factorization':
                predicted_rating = self.predict_matrix_factorization(user_id, item_id)
            else:
                predicted_rating = self.global_mean
            
            predictions.append((item_id, predicted_rating))
        
        # 予測評価値でソートして上位N個を返す
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def evaluate_model(self, df, test_size=0.2, methods=['user_based', 'item_based', 'matrix_factorization']):
        """モデル評価"""
        print("モデル評価を実行中...")
        
        # 訓練・テストデータ分割
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # 訓練データでモデル構築
        train_matrix = self.create_user_item_matrix(train_df)
        
        if 'user_based' in methods:
            self.calculate_user_similarity()
        if 'item_based' in methods:
            self.calculate_item_similarity()
        if 'matrix_factorization' in methods:
            self.matrix_factorization()
        
        # テストデータで予測・評価
        evaluation_results = {}
        
        for method in methods:
            predictions = []
            actuals = []
            
            for _, row in test_df.iterrows():
                user_id = row['user_id']
                product_id = row['product_id']
                actual_rating = row['rating']
                
                if method == 'user_based':
                    predicted_rating = self.predict_user_based(user_id, product_id)
                elif method == 'item_based':
                    predicted_rating = self.predict_item_based(user_id, product_id)
                elif method == 'matrix_factorization':
                    predicted_rating = self.predict_matrix_factorization(user_id, product_id)
                
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            
            # 評価指標計算
            mse = mean_squared_error(actuals, predictions)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
            
            evaluation_results[method] = {
                'RMSE': rmse,
                'MAE': mae,
                'MSE': mse
            }
            
            print(f"{method} - RMSE: {rmse:.3f}, MAE: {mae:.3f}")
        
        return evaluation_results
    
    def visualize_recommendations(self):
        """推薦システムの可視化"""
        if self.user_item_matrix is None:
            print("可視化するデータがありません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. ユーザー-アイテム行列のヒートマップ（サンプル）
        ax1 = axes[0, 0]
        sample_matrix = self.user_item_matrix.iloc[:20, :20]  # 20x20のサンプル
        sns.heatmap(sample_matrix, ax=ax1, cmap='YlOrRd', cbar_kws={'label': 'Rating'})
        ax1.set_title('User-Item Matrix (Sample)')
        ax1.set_xlabel('Products')
        ax1.set_ylabel('Users')
        
        # 2. 評価値分布
        ax2 = axes[0, 1]
        ratings = self.user_item_matrix.values.flatten()
        ratings_nonzero = ratings[ratings > 0]
        ax2.hist(ratings_nonzero, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Rating Distribution')
        ax2.set_xlabel('Rating')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # 3. ユーザー類似度分布（上位類似度のみ）
        if self.user_similarity is not None:
            ax3 = axes[1, 0]
            # 対角要素（自分自身）を除いた類似度
            similarity_values = []
            for i in range(len(self.user_similarity)):
                for j in range(i+1, len(self.user_similarity)):
                    similarity_values.append(self.user_similarity.iloc[i, j])
            
            similarity_values = [s for s in similarity_values if s > 0]  # 正の類似度のみ
            
            if len(similarity_values) > 0:
                ax3.hist(similarity_values, bins=30, alpha=0.7, color='lightgreen')
                ax3.set_title('User Similarity Distribution')
                ax3.set_xlabel('Cosine Similarity')
                ax3.set_ylabel('Count')
                ax3.grid(True, alpha=0.3)
        
        # 4. アイテム類似度分布
        if self.item_similarity is not None:
            ax4 = axes[1, 1]
            # 対角要素を除いた類似度
            similarity_values = []
            for i in range(len(self.item_similarity)):
                for j in range(i+1, len(self.item_similarity)):
                    similarity_values.append(self.item_similarity.iloc[i, j])
            
            similarity_values = [s for s in similarity_values if s > 0]  # 正の類似度のみ
            
            if len(similarity_values) > 0:
                ax4.hist(similarity_values, bins=30, alpha=0.7, color='lightcoral')
                ax4.set_title('Item Similarity Distribution')
                ax4.set_xlabel('Cosine Similarity')
                ax4.set_ylabel('Count')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_dir = Path("results/recommendation")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "collaborative_filtering_analysis.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_recommendations(self, user_recommendations, filename="recommendations.csv"):
        """推薦結果の保存"""
        output_dir = Path("results/recommendation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 推薦結果をDataFrameに変換
        recommendation_data = []
        
        for user_id, recommendations in user_recommendations.items():
            for rank, (item_id, predicted_rating) in enumerate(recommendations, 1):
                recommendation_data.append({
                    'user_id': user_id,
                    'product_id': item_id,
                    'predicted_rating': predicted_rating,
                    'recommendation_rank': rank
                })
        
        recommendations_df = pd.DataFrame(recommendation_data)
        
        # 保存
        output_file = output_dir / filename
        recommendations_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"推薦結果を保存: {output_file}")
        
        return output_file

def create_sample_interaction_data():
    """サンプル相互作用データの生成"""
    np.random.seed(42)
    
    # ユーザーと商品の数
    n_users = 200
    n_products = 100
    n_interactions = 5000
    
    users = [f"USER_{i:04d}" for i in range(1, n_users + 1)]
    products = [f"PROD_{i:04d}" for i in range(1, n_products + 1)]
    
    interaction_data = []
    
    # ユーザーの嗜好パターンを作成
    user_preferences = {}
    for user in users:
        # 各ユーザーは特定の商品カテゴリを好む傾向
        preferred_categories = np.random.choice(5, size=np.random.randint(1, 4), replace=False)
        user_preferences[user] = preferred_categories
    
    # 商品のカテゴリを設定
    product_categories = {}
    for product in products:
        product_categories[product] = np.random.randint(0, 5)
    
    # 相互作用データ生成
    for _ in range(n_interactions):
        user = np.random.choice(users)
        product = np.random.choice(products)
        
        # ユーザーの嗜好に基づいて評価値を調整
        if product_categories[product] in user_preferences[user]:
            # 好みのカテゴリは高評価傾向
            rating = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
        else:
            # 好みでないカテゴリは低評価傾向
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        
        interaction_data.append({
            'user_id': user,
            'product_id': product,
            'rating': rating,
            'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
        })
    
    # 重複を除去
    df = pd.DataFrame(interaction_data)
    df = df.drop_duplicates(subset=['user_id', 'product_id']).reset_index(drop=True)
    
    # 保存
    output_dir = Path("data/sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "user_interactions.csv", index=False, encoding='utf-8-sig')
    print(f"サンプル相互作用データを生成: {output_dir / 'user_interactions.csv'}")
    print(f"  最終データ数: {len(df)}件")
    
    return df

def main():
    """メイン実行関数"""
    print("協調フィルタリング推薦システムを開始します...")
    
    # サンプルデータ生成
    if not Path("data/sample_data/user_interactions.csv").exists():
        print("サンプル相互作用データを生成中...")
        create_sample_interaction_data()
    
    # 推薦システムの初期化
    recommender = CollaborativeFilteringRecommender()
    
    # データ読み込み
    df = recommender.load_interaction_data("data/sample_data/user_interactions.csv")
    if df is None:
        return
    
    # ユーザー-アイテム行列作成
    user_item_matrix = recommender.create_user_item_matrix(df)
    
    # 類似度計算
    print("\nユーザー間類似度を計算中...")
    recommender.calculate_user_similarity()
    
    print("アイテム間類似度を計算中...")
    recommender.calculate_item_similarity()
    
    print("行列分解を実行中...")
    recommender.matrix_factorization()
    
    # モデル評価
    print("\nモデル評価を実行中...")
    evaluation_results = recommender.evaluate_model(df)
    
    # サンプルユーザーに対する推薦
    sample_users = df['user_id'].unique()[:5]
    user_recommendations = {}
    
    print("\nサンプルユーザーへの推薦を実行中...")
    for user_id in sample_users:
        recommendations = recommender.recommend_items(user_id, n_recommendations=10, method='item_based')
        user_recommendations[user_id] = recommendations
        
        print(f"\nユーザー {user_id} への推薦:")
        for i, (product_id, rating) in enumerate(recommendations[:5], 1):
            print(f"  {i}. {product_id} (予測評価: {rating:.2f})")
    
    # 可視化
    recommender.visualize_recommendations()
    
    # 結果保存
    recommender.save_recommendations(user_recommendations)
    
    # 最終レポート
    print(f"\n=== 協調フィルタリング推薦システム レポート ===")
    print(f"データ統計:")
    print(f"  ユーザー数: {df['user_id'].nunique()}")
    print(f"  商品数: {df['product_id'].nunique()}")
    print(f"  評価数: {len(df)}")
    
    print(f"\nモデル性能:")
    for method, metrics in evaluation_results.items():
        print(f"  {method}:")
        print(f"    RMSE: {metrics['RMSE']:.3f}")
        print(f"    MAE: {metrics['MAE']:.3f}")
    
    # 最良モデルの特定
    best_method = min(evaluation_results.keys(), key=lambda x: evaluation_results[x]['RMSE'])
    print(f"\n🏆 最良モデル: {best_method} (RMSE: {evaluation_results[best_method]['RMSE']:.3f})")
    
    print(f"\n✅ 協調フィルタリング推薦システムが完了しました！")
    print("📊 results/recommendation/ フォルダに結果が保存されています。")

if __name__ == "__main__":
    main()
