"""
時系列分析による需要予測モデル
ARIMA、指数平滑法、季節分解などの時系列手法を実装
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    def __init__(self):
        """時系列予測クラスの初期化"""
        self.models = {}
        self.forecast_results = {}
        
    def load_and_prepare_data(self, file_path: str, product_id: str = None):
        """データの読み込みと前処理"""
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            
            if product_id:
                df = df[df['product_id'] == product_id]
            
            # 日次データに変換
            df = df.set_index('date').resample('D')['quantity'].sum().fillna(0)
            
            print(f"データ準備完了: {len(df)}日分")
            return df
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return None
    
    def check_stationarity(self, ts_data, title="Time Series"):
        """定常性の確認（ADF検定）"""
        result = adfuller(ts_data.dropna())
        
        print(f"\n=== {title} の定常性検定結果 ===")
        print(f'ADF統計量: {result[0]:.6f}')
        print(f'p値: {result[1]:.6f}')
        print(f'臨界値:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.3f}')
        
        if result[1] <= 0.05:
            print("結果: 定常である（p < 0.05）")
            return True
        else:
            print("結果: 非定常である（p >= 0.05）")
            return False
    
    def make_stationary(self, ts_data):
        """非定常データの定常化"""
        # 1次差分
        diff_data = ts_data.diff().dropna()
        
        if self.check_stationarity(diff_data, "1次差分"):
            return diff_data, 1
        
        # 2次差分
        diff2_data = diff_data.diff().dropna()
        if self.check_stationarity(diff2_data, "2次差分"):
            return diff2_data, 2
        
        # 対数変換 + 差分
        log_data = np.log1p(ts_data)
        log_diff_data = log_data.diff().dropna()
        
        if self.check_stationarity(log_diff_data, "対数差分"):
            return log_diff_data, "log_diff"
        
        return diff_data, 1  # デフォルトは1次差分
    
    def seasonal_decomposition(self, ts_data, period=7):
        """季節分解の実行"""
        try:
            if len(ts_data) < 2 * period:
                print("データが不足しているため季節分解をスキップします")
                return None
            
            decomposition = seasonal_decompose(
                ts_data, 
                model='additive', 
                period=period
            )
            
            # 可視化
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            decomposition.observed.plot(ax=axes[0], title='Original')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')
            
            plt.tight_layout()
            
            # 保存
            output_dir = Path("results/forecasting")
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / "seasonal_decomposition.png", dpi=300, bbox_inches='tight')
            
            return decomposition
            
        except Exception as e:
            print(f"季節分解エラー: {e}")
            return None
    
    def auto_arima_selection(self, ts_data, max_p=5, max_d=2, max_q=5):
        """ARIMA パラメータの自動選択（AIC基準）"""
        best_aic = float('inf')
        best_params = None
        best_model = None
        
        print("ARIMA パラメータを最適化中...")
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(ts_data, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                            best_model = fitted_model
                            
                    except Exception:
                        continue
        
        print(f"最適ARIMA パラメータ: {best_params}, AIC: {best_aic:.2f}")
        return best_model, best_params
    
    def fit_arima_model(self, ts_data, order=None):
        """ARIMA モデルの学習"""
        try:
            if order is None:
                model, params = self.auto_arima_selection(ts_data)
            else:
                arima_model = ARIMA(ts_data, order=order)
                model = arima_model.fit()
                params = order
            
            self.models['arima'] = {
                'model': model,
                'params': params,
                'aic': model.aic
            }
            
            print(f"ARIMA{params} モデル学習完了 (AIC: {model.aic:.2f})")
            return model
            
        except Exception as e:
            print(f"ARIMA モデル学習エラー: {e}")
            return None
    
    def fit_exponential_smoothing(self, ts_data, seasonal_periods=7):
        """指数平滑法モデルの学習"""
        try:
            # シンプル指数平滑法
            simple_model = ExponentialSmoothing(ts_data, trend=None, seasonal=None)
            simple_fit = simple_model.fit()
            
            # Holt法（トレンド考慮）
            holt_model = ExponentialSmoothing(ts_data, trend='add', seasonal=None)
            holt_fit = holt_model.fit()
            
            # Holt-Winters法（季節性考慮）
            if len(ts_data) >= 2 * seasonal_periods:
                hw_model = ExponentialSmoothing(
                    ts_data, 
                    trend='add', 
                    seasonal='add', 
                    seasonal_periods=seasonal_periods
                )
                hw_fit = hw_model.fit()
            else:
                hw_fit = holt_fit  # データ不足の場合はHolt法を使用
            
            # AIC で最適モデルを選択
            models_to_compare = {
                'simple': simple_fit,
                'holt': holt_fit,
                'holt_winters': hw_fit
            }
            
            best_model = min(models_to_compare.items(), key=lambda x: x[1].aic)
            
            self.models['exponential_smoothing'] = {
                'model': best_model[1],
                'type': best_model[0],
                'aic': best_model[1].aic
            }
            
            print(f"指数平滑法 ({best_model[0]}) モデル学習完了 (AIC: {best_model[1].aic:.2f})")
            return best_model[1]
            
        except Exception as e:
            print(f"指数平滑法モデル学習エラー: {e}")
            return None
    
    def generate_forecasts(self, steps=30):
        """予測の実行"""
        forecasts = {}
        
        for model_name, model_info in self.models.items():
            try:
                model = model_info['model']
                
                if model_name == 'arima':
                    forecast = model.forecast(steps=steps)
                    conf_int = model.get_forecast(steps=steps).conf_int()
                    
                elif model_name == 'exponential_smoothing':
                    forecast = model.forecast(steps=steps)
                    # 信頼区間の簡易計算
                    residuals_std = np.std(model.resid)
                    conf_int = pd.DataFrame({
                        'lower': forecast - 1.96 * residuals_std,
                        'upper': forecast + 1.96 * residuals_std
                    })
                
                forecasts[model_name] = {
                    'forecast': forecast,
                    'conf_int': conf_int,
                    'model_type': model_name
                }
                
                print(f"{model_name} 予測完了: {steps}日分")
                
            except Exception as e:
                print(f"{model_name} 予測エラー: {e}")
        
        self.forecast_results = forecasts
        return forecasts
    
    def evaluate_models(self, ts_data, test_size=30):
        """モデル性能の評価"""
        if len(ts_data) < test_size * 2:
            print("評価用データが不足しています")
            return {}
        
        # 訓練・テストデータ分割
        train_data = ts_data[:-test_size]
        test_data = ts_data[-test_size:]
        
        evaluation_results = {}
        
        # 各モデルで予測・評価
        for model_name in self.models.keys():
            try:
                if model_name == 'arima':
                    # ARIMA モデルで予測
                    arima_model = ARIMA(train_data, order=self.models[model_name]['params'])
                    fitted_model = arima_model.fit()
                    forecast = fitted_model.forecast(steps=test_size)
                    
                elif model_name == 'exponential_smoothing':
                    # 指数平滑法で予測
                    if self.models[model_name]['type'] == 'simple':
                        es_model = ExponentialSmoothing(train_data, trend=None, seasonal=None)
                    elif self.models[model_name]['type'] == 'holt':
                        es_model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
                    else:  # holt_winters
                        es_model = ExponentialSmoothing(
                            train_data, trend='add', seasonal='add', seasonal_periods=7
                        )
                    
                    fitted_model = es_model.fit()
                    forecast = fitted_model.forecast(steps=test_size)
                
                # 評価指標計算
                mae = mean_absolute_error(test_data, forecast)
                rmse = np.sqrt(mean_squared_error(test_data, forecast))
                mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
                
                evaluation_results[model_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'forecast': forecast,
                    'actual': test_data
                }
                
                print(f"{model_name} 評価完了 - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
                
            except Exception as e:
                print(f"{model_name} 評価エラー: {e}")
        
        return evaluation_results
    
    def visualize_forecasts(self, ts_data, evaluation_results=None):
        """予測結果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. 全体の時系列データ
        ax1 = axes[0, 0]
        ts_data.plot(ax=ax1, label='実績値', color='blue', alpha=0.7)
        
        # 予測結果をプロット
        if self.forecast_results:
            start_date = ts_data.index[-1]
            
            for model_name, forecast_data in self.forecast_results.items():
                forecast = forecast_data['forecast']
                future_dates = pd.date_range(
                    start=start_date + pd.Timedelta(days=1), 
                    periods=len(forecast), 
                    freq='D'
                )
                
                color = {'arima': 'red', 'exponential_smoothing': 'green'}.get(model_name, 'orange')
                
                ax1.plot(future_dates, forecast, label=f'{model_name} 予測', 
                        color=color, linestyle='--', linewidth=2)
                
                # 信頼区間
                conf_int = forecast_data['conf_int']
                ax1.fill_between(future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                               color=color, alpha=0.2)
        
        ax1.set_title('時系列データと予測結果')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 季節性パターン
        ax2 = axes[0, 1]
        if len(ts_data) > 14:
            # 曜日別の平均需要
            ts_df = ts_data.to_frame('quantity')
            ts_df['weekday'] = ts_df.index.dayofweek
            weekday_avg = ts_df.groupby('weekday')['quantity'].mean()
            
            weekday_names = ['月', '火', '水', '木', '金', '土', '日']
            ax2.bar(weekday_names, weekday_avg.values, color='lightcoral', alpha=0.7)
            ax2.set_title('曜日別平均需要')
            ax2.set_ylabel('平均需要')
        
        # 3. モデル比較（評価結果がある場合）
        ax3 = axes[1, 0]
        if evaluation_results:
            models = list(evaluation_results.keys())
            mae_scores = [evaluation_results[m]['MAE'] for m in models]
            mape_scores = [evaluation_results[m]['MAPE'] for m in models]
            
            x = range(len(models))
            width = 0.35
            
            ax3_twin = ax3.twinx()
            
            bars1 = ax3.bar([i - width/2 for i in x], mae_scores, width, 
                           label='MAE', color='skyblue', alpha=0.7)
            bars2 = ax3_twin.bar([i + width/2 for i in x], mape_scores, width, 
                                label='MAPE (%)', color='lightgreen', alpha=0.7)
            
            ax3.set_xlabel('モデル')
            ax3.set_ylabel('MAE')
            ax3_twin.set_ylabel('MAPE (%)')
            ax3.set_title('モデル性能比較')
            ax3.set_xticks(x)
            ax3.set_xticklabels(models)
            
            # 数値ラベル追加
            for bar, value in zip(bars1, mae_scores):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}', ha='center', va='bottom')
            
            for bar, value in zip(bars2, mape_scores):
                ax3_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                             f'{value:.1f}%', ha='center', va='bottom')
        
        # 4. 残差分析
        ax4 = axes[1, 1]
        if evaluation_results:
            for model_name, results in evaluation_results.items():
                residuals = results['actual'] - results['forecast']
                ax4.hist(residuals, bins=15, alpha=0.5, label=f'{model_name} 残差')
            
            ax4.set_title('予測残差分布')
            ax4.set_xlabel('残差')
            ax4.set_ylabel('頻度')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_dir = Path("results/forecasting")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "forecast_analysis.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_forecast_results(self):
        """予測結果の保存"""
        if not self.forecast_results:
            print("保存する予測結果がありません")
            return
        
        output_dir = Path("results/forecasting")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 各モデルの予測結果を保存
        for model_name, forecast_data in self.forecast_results.items():
            forecast_df = pd.DataFrame({
                'forecast': forecast_data['forecast'],
                'lower_bound': forecast_data['conf_int'].iloc[:, 0],
                'upper_bound': forecast_data['conf_int'].iloc[:, 1]
            })
            
            # 日付インデックスを追加
            start_date = pd.Timestamp.now().date()
            forecast_df['date'] = pd.date_range(
                start=start_date, 
                periods=len(forecast_df), 
                freq='D'
            )
            
            # 保存
            filename = output_dir / f"{model_name}_forecast.csv"
            forecast_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"{model_name} 予測結果を保存: {filename}")
        
        return output_dir

def create_sample_time_series_data():
    """サンプル時系列データの生成"""
    np.random.seed(42)
    
    # 1年分の日次データ
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # ベース需要
    base_demand = 50
    
    # トレンド（年間で20%増加）
    trend = np.linspace(0, base_demand * 0.2, len(dates))
    
    # 季節性（週次・月次）
    weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    monthly_seasonality = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
    
    # ノイズ
    noise = np.random.normal(0, 5, len(dates))
    
    # 最終的な需要データ
    demand = base_demand + trend + weekly_seasonality + monthly_seasonality + noise
    demand = np.maximum(demand, 0)  # 負の値を0にクリップ
    
    # データフレーム作成
    ts_data = pd.DataFrame({
        'date': dates,
        'product_id': 'PROD_0001',
        'quantity': demand.astype(int)
    })
    
    # 保存
    output_dir = Path("data/sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_data.to_csv(output_dir / "time_series_demand.csv", index=False, encoding='utf-8-sig')
    print(f"サンプル時系列データを生成: {output_dir / 'time_series_demand.csv'}")
    
    return ts_data

def main():
    """メイン実行関数"""
    print("時系列需要予測を開始します...")
    
    # サンプルデータ生成
    if not Path("data/sample_data/time_series_demand.csv").exists():
        print("サンプル時系列データを生成中...")
        create_sample_time_series_data()
    
    # 予測器の初期化
    forecaster = TimeSeriesForecaster()
    
    # データ読み込み
    ts_data = forecaster.load_and_prepare_data(
        "data/sample_data/time_series_demand.csv", 
        "PROD_0001"
    )
    
    if ts_data is None:
        return
    
    # 季節分解
    decomposition = forecaster.seasonal_decomposition(ts_data)
    
    # 定常性チェック
    is_stationary = forecaster.check_stationarity(ts_data)
    
    # モデル学習
    print("\nARIMAモデルを学習中...")
    arima_model = forecaster.fit_arima_model(ts_data)
    
    print("\n指数平滑法モデルを学習中...")
    es_model = forecaster.fit_exponential_smoothing(ts_data)
    
    # 予測実行
    print("\n予測を実行中...")
    forecasts = forecaster.generate_forecasts(steps=30)
    
    # モデル評価
    print("\nモデル性能を評価中...")
    evaluation = forecaster.evaluate_models(ts_data)
    
    # 可視化
    forecaster.visualize_forecasts(ts_data, evaluation)
    
    # 結果保存
    forecaster.save_forecast_results()
    
    # 最適モデルの推奨
    if evaluation:
        best_model = min(evaluation.items(), key=lambda x: x[1]['MAPE'])
        print(f"\n🏆 推奨モデル: {best_model[0]} (MAPE: {best_model[1]['MAPE']:.2f}%)")
    
    print("\n✅ 時系列需要予測が完了しました！")
    print("📊 results/forecasting/ フォルダに結果が保存されています。")

if __name__ == "__main__":
    main()
