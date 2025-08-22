# 2. 予知保全AI - 異常検知システム
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

# センサーデータ前処理
def preprocess_sensor_data(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# 異常検知モデル
def create_anomaly_detector():
    model = IsolationForest(
        contamination=0.1,  # 異常データ10%想定
        random_state=42,
        n_estimators=100
    )
    return model

# リアルタイム監視
def real_time_monitoring(sensor_data, model, threshold=-0.1):
    anomaly_score = model.decision_function([sensor_data])
    if anomaly_score[0] < threshold:
        return "🚨 異常検知！メンテナンス推奨"
    return "✅ 正常運転中"

# 使用例
detector = create_anomaly_detector()
detector.fit(normal_sensor_data)
result = real_time_monitoring(current_data, detector)
print(result)
