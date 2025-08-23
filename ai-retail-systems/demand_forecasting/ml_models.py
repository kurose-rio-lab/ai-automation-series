"""
機械学習による需要予測モデル
Random Forest、XGBoost、LSTMなどの機械学習手法を実装
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MLDemandForecaster:
    def __init__(self):
        """機械学習需要予測クラスの初期化"""
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'quantity'
        
    def load_and_prepare_data(self, file_path: str):
        """データの読み込みと前処理"""
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            
            print(f"データ読み込み完了: {len(df)}レコード")
            return df
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return None
    
    def create_features(self, df):
        """特徴量エンジニアリング"""
        print("特徴量を作成中...")
        
        # データフレームをコピー
        feature_df = df.copy()
        
        # 基本的な時間特徴量
        feature_df['year'] = feature_df['date'].dt.year
        feature_df['month'] = feature_df['date'].dt.month
        feature_df['day'] = feature_df['date'].dt.day
        feature_df['dayofweek'] = feature_df['date'].dt.dayofweek
        feature_df['dayofyear'] = feature_df['date'].dt.dayofyear
        feature_df['quarter'] = feature_df['date'].dt.quarter
        
        # 週末フラグ
        feature_df['is_weekend'] = (feature_df['dayofweek'] >= 5).astype(int)
        
        # 月の始まり・終わりフラグ
        feature_df['is_month_start'] = (feature_df['day'] <= 5).astype(int)
        feature_df['is_month_end'] = (feature_df['day'] >= 25).astype(int)
        
        # 季節性特徴量（サイクリック）
        feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / 12)
        feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / 12)
        feature_df['dayofweek_sin'] = np.sin(2 * np.pi * feature_df['dayofweek'] / 7)
        feature_df['dayofweek_cos'] = np.cos(2 * np.pi * feature_df['dayofweek'] / 7)
        
        # 商品別特徴量を作成するためにソート
        feature_df = feature_df.sort_values(['product_id', 'date'])
        
        # ラグ特徴量（過去の値）
        lag_periods = [1, 2, 3, 7, 14, 30]
        for lag in lag_periods:
            feature_df[f'quantity_lag_{lag}'] = feature_df.groupby('product_id')['quantity'].shift(lag)
        
        # 移動平均特徴量
        window_sizes = [3, 7, 14, 30]
        for window in window_sizes:
            feature_df[f'quantity_ma_{window}'] = (
                feature_df.groupby('product_id')['quantity']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        
        # 移動標準偏差（変動性）
        for window in [7, 14, 30]:
            feature_df[f'quantity_std_{window}'] = (
                feature_df.groupby('product_id')['quantity']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
        
        # 指数移動平均
        for alpha in [0.1, 0.3, 0.5]:
            feature_df[f'quantity_ema_{alpha}'] = (
                feature_df.groupby('product_id')['quantity']
                .ewm(alpha=alpha)
                .mean()
                .reset_index(level=0, drop=True)
            )
        
        # トレンド特徴量
        feature_df['quantity_trend'] = (
            feature_df.groupby('product_id')['quantity']
            .pct_change()
            .fillna(0)
        )
        
        # 商品IDをエンコード
        le = LabelEncoder()
        feature_df['product_id_encoded'] = le.fit_transform(feature_df['product_id'])
        
        # 外部特徴量（祝日、イベントなど）
        feature_df = self.add_external_features(feature_df)
        
        # 欠損値を埋める
        feature_df = feature_df.fillna(method='ffill').fillna(0)
        
        print(f"特徴量作成完了: {feature_df.shape[1]}次元")
        
        return feature_df
    
    def add_external_features(self, df):
        """外部特徴量の追加"""
        # 日本の祝日（簡易版）
        holidays = [
            '2023-01-01', '2023-01-09', '2023-02-11', '2023-02-23',
            '2023-03-21', '2023-04-29', '2023-05-03', '2023-05-04',
            '2023-05-05', '2023-07-17', '2023-08-11', '2023-09-18',
            '2023-09-23', '2023-10-09', '2023-11-03', '2023-11-23',
            '2023-12-29', '2023-12-30', '2023-12-31'
        ]
        
        holiday_dates = pd.to_datetime(holidays)
        df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)
        
        # 大型連休フラグ
        golden_week = pd.date_range('2023-04-29', '2023-05-07')
        summer_vacation = pd.date_range('2023-08-11', '2023-08-20')
        year_end = pd.date_range('2023-12-29', '2023-12-31')
        
        df['is_golden_week'] = df['date'].isin(golden_week).astype(int)
        df['is_summer_vacation'] = df['date'].isin(summer_vacation).astype(int)
        df['is_year_end'] = df['date'].isin(year_end).astype(int)
        
        # セール期間（仮想的）
        spring_sale = pd.date_range('2023-03-01', '2023-03-31')
        summer_sale = pd.date_range('2023-07-01', '2023-07-31')
        winter_sale = pd.date_range('2023-11-01', '2023-11-30')
        
        df['is_sale_period'] = (
            df['date'].isin(spring_sale) | 
            df['date'].isin(summer_sale) | 
            df['date'].isin(winter_sale)
        ).astype(int)
        
        # 天候情報（簡易シミュレーション）
        np.random.seed(42)
        df['temperature'] = 20 + 10 * np.sin(2 * np.pi * df['dayofyear'] / 365) + np.random.normal(0, 3, len(df))
        df['is_rainy'] = (np.random.random(len(df)) < 0.3).astype(int)
        
        return df
    
    def prepare_ml_data(self, feature_df, target_col='quantity'):
        """機械学習用のデータ準備"""
        # 特徴量列を選択（日付、商品ID、ターゲット以外）
        exclude_columns = ['date', 'product_id', target_col]
        feature_columns = [col for col in feature_df.columns if col not in exclude_columns]
        
        # 特徴量とターゲット
        X = feature_df[feature_columns]
        y = feature_df[target_col]
        
        # 欠損値がある行を除外
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        self.feature_columns = feature_columns
        
        print(f"ML用データ準備完了: {X.shape[0]}サンプル, {X.shape[1]}特徴量")
        
        return X, y
    
    def train_random_forest(self, X, y, test_size=0.2):
        """Random Forest モデルの訓練"""
        print("Random Forest モデルを訓練中...")
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        # ハイパーパラメータ最適化
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # グリッドサーチ（時間短縮のため小さなサンプルで）
        if len(X_train) > 10000:
            sample_indices = np.random.choice(len(X_train), 5000, replace=False)
            X_sample = X_train.iloc[sample_indices]
            y_sample = y_train.iloc[sample_indices]
        else:
            X_sample = X_train
            y_sample = y_train
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_sample, y_sample)
        best_rf = grid_search.best_estimator_
        
        # 全データで再訓練
        best_rf.fit(X_train, y_train)
        
        # 予測と評価
        y_pred = best_rf.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        self.models['random_forest'] = {
            'model': best_rf,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': dict(zip(self.feature_columns, best_rf.feature_importances_))
        }
        
        print(f"Random Forest 訓練完了 - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        return best_rf, X_test, y_test, y_pred
    
    def train_xgboost(self, X, y, test_size=0.2):
        """XGBoost モデルの訓練"""
        print("XGBoost モデルを訓練中...")
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # XGBoost パラメータ
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # モデル訓練
        xgb_model = xgb.XGBRegressor(**params)
        
        # Early stopping
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # 予測と評価
        y_pred = xgb_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # 特徴量重要度
        feature_importance = dict(zip(self.feature_columns, xgb_model.feature_importances_))
        
        self.models['xgboost'] = {
            'model': xgb_model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': feature_importance
        }
        
        print(f"XGBoost 訓練完了 - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        return xgb_model, X_test, y_test, y_pred
    
    def prepare_lstm_data(self, df, lookback_days=30, product_id=None):
        """LSTM用のデータ準備"""
        if product_id:
            df = df[df['product_id'] == product_id].copy()
        
        # 日次データにリサンプル
        df = df.set_index('date')['quantity'].resample('D').sum().fillna(0)
        
        # 正規化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.values.reshape(-1, 1)).flatten()
        
        # シーケンスデータ作成
        X, y = [], []
        for i in range(lookback_days, len(scaled_data)):
            X.append(scaled_data[i-lookback_days:i])
            y.append(scaled_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # LSTM用に次元変更 (samples, time steps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        self.scalers['lstm'] = scaler
        
        return X, y, scaler
    
    def train_lstm(self, df, lookback_days=30, product_id='PROD_0001'):
        """LSTM モデルの訓練"""
        print(f"LSTM モデルを訓練中 (商品ID: {product_id})...")
        
        # データ準備
        X, y, scaler = self.prepare_lstm_data(df, lookback_days, product_id)
        
        if len(X) < 100:
            print("LSTM訓練用のデータが不足しています")
            return None, None, None, None
        
        # データ分割
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # LSTM モデル構築
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback_days, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # モデル訓練
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        # 予測と評価
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # 正規化を元に戻す
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = scaler.inverse_transform(y_pred_scaled).flatten()
        
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        r2 = r2_score(y_test_original, y_pred_original)
        
        self.models['lstm'] = {
            'model': model,
            'scaler': scaler,
            'lookback_days': lookback_days,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'history': history
        }
        
        print(f"LSTM 訓練完了 - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        return model, X_test, y_test_original, y_pred_original
    
    def visualize_feature_importance(self):
        """特徴量重要度の可視化"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Random Forest の特徴量重要度
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest']['feature_importance']
            top_features = dict(sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:15])
            
            ax1 = axes[0]
            bars = ax1.barh(list(top_features.keys()), list(top_features.values()), color='skyblue')
            ax1.set_title('Random Forest 特徴量重要度 (Top 15)')
            ax1.set_xlabel('重要度')
            
            # 数値ラベル追加
            for bar, value in zip(bars, top_features.values()):
                ax1.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', va='center', ha='left')
        
        # XGBoost の特徴量重要度
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost']['feature_importance']
            top_features = dict(sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)[:15])
            
            ax2 = axes[1]
            bars = ax2.barh(list(top_features.keys()), list(top_features.values()), color='lightcoral')
            ax2.set_title('XGBoost 特徴量重要度 (Top 15)')
            ax2.set_xlabel('重要度')
            
            # 数値ラベル追加
            for bar, value in zip(bars, top_features.values()):
                ax2.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        
        # 保存
        output_dir = Path("results/ml_forecasting")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_model_comparison(self):
        """モデル性能比較の可視化"""
        if not self.models:
            print("比較するモデルがありません")
            return
        
        # 性能指標をまとめる
        model_names = list(self.models.keys())
        mae_scores = [self.models[name]['mae'] for name in model_names]
        rmse_scores = [self.models[name]['rmse'] for name in model_names]
        r2_scores = [self.models[name]['r2'] for name in model_names]
        
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # MAE比較
        ax1 = axes[0]
        bars1 = ax1.bar(model_names, mae_scores, color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
        ax1.set_title('平均絶対誤差 (MAE) 比較')
        ax1.set_ylabel('MAE')
        
        for bar, value in zip(bars1, mae_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # RMSE比較
        ax2 = axes[1]
        bars2 = ax2.bar(model_names, rmse_scores, color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
        ax2.set_title('平方根平均二乗誤差 (RMSE) 比較')
        ax2.set_ylabel('RMSE')
        
        for bar, value in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # R²比較
        ax3 = axes[2]
        bars3 = ax3.bar(model_names, r2_scores, color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
        ax3.set_title('決定係数 (R²) 比較')
        ax3.set_ylabel('R²')
        ax3.set_ylim(0, 1)
        
        for bar, value in zip(bars3, r2_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存
        output_dir = Path("results/ml_forecasting")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_models(self):
        """訓練済みモデルの保存"""
        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        
        for model_name, model_info in self.models.items():
            if model_name == 'lstm':
                # Kerasモデルの保存
                model_info['model'].save(output_dir / f"{model_name}_model.keras")
                # スケーラーの保存
                joblib.dump(model_info['scaler'], output_dir / f"{model_name}_scaler.pkl")
            else:
                # scikit-learn、XGBoostモデルの保存
                joblib.dump(model_info['model'], output_dir / f"{model_name}_model.pkl")
        
        # 特徴量列の保存
        joblib.dump(self.feature_columns, output_dir / "feature_columns.pkl")
        
        print(f"モデルを保存: {output_dir}")
    
    def generate_ml_forecast_report(self):
        """機械学習予測レポートの生成"""
        print("\n" + "="*60)
        print("機械学習需要予測 モデル性能レポート")
        print("="*60)
        
        if not self.models:
            print("訓練されたモデルがありません")
            return
        
        # 最良モデルの特定
        best_model_name = min(self.models.keys(), key=lambda x: self.models[x]['mae'])
        best_model = self.models[best_model_name]
        
        print(f"\n🏆 最良モデル: {best_model_name}")
        print(f"   MAE: {best_model['mae']:.2f}")
        print(f"   RMSE: {best_model['rmse']:.2f}")
        print(f"   R²: {best_model['r2']:.3f}")
        
        print(f"\n📊 全モデル性能比較:")
        for model_name, model_info in self.models.items():
            print(f"  {model_name}:")
            print(f"    MAE: {model_info['mae']:.2f}")
            print(f"    RMSE: {model_info['rmse']:.2f}")
            print(f"    R²: {model_info['r2']:.3f}")
        
        # 特徴量重要度（Random Forest/XGBoostのみ）
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest']['feature_importance']
            top_features = dict(sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:5])
            
            print(f"\n🔍 Random Forest 重要特徴量 Top 5:")
            for feature, importance in top_features.items():
                print(f"  {feature}: {importance:.4f}")
        
        return best_model_name, best_model

def create_comprehensive_demand_data():
    """包括的なサンプル需要データの生成"""
    np.random.seed(42)
    
    # 複数商品、1年分のデータ
    products = [f"PROD_{i:04d}" for i in range(1, 21)]  # 20商品
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    demand_data = []
    
    for product_id in products:
        # 商品ごとの特性
        base_demand = np.random.uniform(20, 100)
        seasonality_strength = np.random.uniform(0.1, 0.3)
        trend_strength = np.random.uniform(-0.1, 0.2)
        noise_level = np.random.uniform(0.1, 0.4)
        
        for date in dates:
            day_of_year = date.timetuple().tm_yday
            
            # 季節性
            seasonal_effect = seasonality_strength * np.sin(2 * np.pi * day_of_year / 365)
            
            # 週次パターン
            weekly_effect = 0.2 * np.sin(2 * np.pi * date.weekday() / 7)
            
            # トレンド
            trend_effect = trend_strength * day_of_year / 365
            
            # 祝日効果
            holiday_effect = 0
            if date.weekday() >= 5:  # 週末
                holiday_effect = 0.3
            
            # 基本需要計算
            expected_demand = base_demand * (
                1 + seasonal_effect + weekly_effect + trend_effect + holiday_effect
            )
            
            # ノイズ追加
            actual_demand = max(0, np.random.normal(expected_demand, expected_demand * noise_level))
            
            demand_data.append({
                'date': date,
                'product_id': product_id,
                'quantity': int(actual_demand)
            })
    
    df = pd.DataFrame(demand_data)
    
    # 保存
    output_dir = Path("data/sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "ml_demand_data.csv", index=False, encoding='utf-8-sig')
    print(f"包括的サンプル需要データを生成: {output_dir / 'ml_demand_data.csv'}")
    
    return df

def main():
    """メイン実行関数"""
    print("機械学習需要予測を開始します...")
    
    # サンプルデータ生成
    if not Path("data/sample_data/ml_demand_data.csv").exists():
        print("サンプルデータを生成中...")
        create_comprehensive_demand_data()
    
    # 予測器の初期化
    forecaster = MLDemandForecaster()
    
    # データ読み込み
    df = forecaster.load_and_prepare_data("data/sample_data/ml_demand_data.csv")
    if df is None:
        return
    
    # 特徴量エンジニアリング
    feature_df = forecaster.create_features(df)
    
    # ML用データ準備
    X, y = forecaster.prepare_ml_data(feature_df)
    
    # Random Forest 訓練
    print("\n" + "="*50)
    rf_model, X_test_rf, y_test_rf, y_pred_rf = forecaster.train_random_forest(X, y)
    
    # XGBoost 訓練
    print("\n" + "="*50)
    xgb_model, X_test_xgb, y_test_xgb, y_pred_xgb = forecaster.train_xgboost(X, y)
    
    # LSTM 訓練
    print("\n" + "="*50)
    lstm_model, X_test_lstm, y_test_lstm, y_pred_lstm = forecaster.train_lstm(df)
    
    # 可視化
    print("\n可視化を作成中...")
    forecaster.visualize_feature_importance()
    forecaster.visualize_model_comparison()
    
    # モデル保存
    forecaster.save_models()
    
    # レポート生成
    best_model_name, best_model = forecaster.generate_ml_forecast_report()
    
    print(f"\n✅ 機械学習需要予測が完了しました！")
    print(f"🏆 推奨モデル: {best_model_name}")
    print("📊 results/ml_forecasting/ フォルダに結果が保存されています。")
    print("💾 models/ フォルダに訓練済みモデルが保存されています。")

if __name__ == "__main__":
    main()
