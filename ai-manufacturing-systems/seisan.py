# 3. 生産最適化AI - 需要予測システム
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# 特徴量作成
def create_features(df):
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['lag_1_day'] = df['demand'].shift(1)
    df['lag_7_days'] = df['demand'].shift(7)
    df['rolling_mean_7d'] = df['demand'].rolling(7).mean()
    df['rolling_std_7d'] = df['demand'].rolling(7).std()
    return df

# 需要予測モデル
def demand_forecasting_model():
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    return model

# 予測実行
def predict_demand(model, features):
    prediction = model.predict(features)
    return prediction

# 精度評価
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return f"MAE: {mae:.2f}, MAPE: {mape:.2f}%"
