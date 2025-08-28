"""
ポートフォリオ最適化エンジン

このモジュールは、現代ポートフォリオ理論に基づいて、
顧客のリスク許容度と投資目標に応じた最適な資産配分を計算します。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json
from datetime import datetime, timedelta

class PortfolioOptimizer:
    """
    ポートフォリオ最適化エンジンクラス
    
    現代ポートフォリオ理論を用いて、リスクとリターンのバランスを
    最適化した投資ポートフォリオを構築します。
    """
    
    def __init__(self):
        """
        ポートフォリオ最適化エンジンの初期化
        各種資産クラスの期待リターンとリスクパラメータを設定します。
        """
        print("📈 ポートフォリオ最適化エンジンを初期化中...")
        
        # 資産クラスの定義
        # 実際のシステムでは、リアルタイムの市場データから算出
        self.asset_classes = {
            '国内株式': {
                'expected_return': 0.06,    # 期待年率リターン 6%
                'volatility': 0.18,         # 年率ボラティリティ 18%
                'min_weight': 0.0,          # 最小配分比率
                'max_weight': 0.6           # 最大配分比率
            },
            '海外株式': {
                'expected_return': 0.08,    # 期待年率リターン 8%
                'volatility': 0.22,         # 年率ボラティリティ 22%
                'min_weight': 0.0,
                'max_weight': 0.5
            },
            '国内債券': {
                'expected_return': 0.02,    # 期待年率リターン 2%
                'volatility': 0.05,         # 年率ボラティリティ 5%
                'min_weight': 0.0,
                'max_weight': 0.7
            },
            '海外債券': {
                'expected_return': 0.035,   # 期待年率リターン 3.5%
                'volatility': 0.08,         # 年率ボラティリティ 8%
                'min_weight': 0.0,
                'max_weight': 0.4
            },
            'REIT': {
                'expected_return': 0.055,   # 期待年率リターン 5.5%
                'volatility': 0.15,         # 年率ボラティリティ 15%
                'min_weight': 0.0,
                'max_weight': 0.3
            },
            'コモディティ': {
                'expected_return': 0.04,    # 期待年率リターン 4%
                'volatility': 0.25,         # 年率ボラティリティ 25%
                'min_weight': 0.0,
                'max_weight': 0.2
            }
        }
        
        # 資産間の相関係数行列
        # 実際のシステムでは過去データから計算
        self.correlation_matrix = np.array([
            #    国内株式, 海外株式, 国内債券, 海外債券, REIT, コモディティ
            [1.00, 0.70, 0.10, 0.15, 0.60, 0.30],  # 国内株式
            [0.70, 1.00, 0.05, 0.20, 0.65, 0.35],  # 海外株式
            [0.10, 0.05, 1.00, 0.40, 0.20, -0.10], # 国内債券
            [0.15, 0.20, 0.40, 1.00, 0.25, 0.10],  # 海外債券
            [0.60, 0.65, 0.20, 0.25, 1.00, 0.40],  # REIT
            [0.30, 0.35, -0.10, 0.10, 0.40, 1.00]  # コモディティ
        ])
        
        # 資産名のリスト（順序を保持）
        self.asset_names = list(self.asset_classes.keys())
        
        print("✅ ポートフォリオ最適化エンジン初期化完了")
    
    def optimize_portfolio(self, customer_info, risk_score):
        """
        ポートフォリオの最適化実行
        
        顧客情報とリスクスコアに基づいて、最適な資産配分を計算します。
        
        Args:
            customer_info (dict): 顧客情報
            risk_score (float): リスクスコア (0-100)
            
        Returns:
            dict: 最適化結果
        """
        print("🔍 ポートフォリオ最適化を実行中...")
        
        # リスク許容度の決定
        risk_tolerance = self.determine_risk_tolerance(risk_score, customer_info['age'])
        
        # 最適化の実行
        optimal_weights = self.calculate_optimal_weights(risk_tolerance)
        
        # 期待リターンとリスクの計算
        expected_return = self.calculate_portfolio_return(optimal_weights)
        portfolio_risk = self.calculate_portfolio_risk(optimal_weights)
        
        # シャープレシオの計算（リスクフリーレート2%と仮定）
        risk_free_rate = 0.02
        sharpe_ratio = (expected_return - risk_free_rate) / portfolio_risk
        
        # 投資額別の配分金額計算（サンプル投資額を設定）
        investment_amount = customer_info.get('investment_amount', 1000000)  # デフォルト100万円
        allocation_amounts = {
            asset: weight * investment_amount 
            for asset, weight in zip(self.asset_names, optimal_weights)
        }
        
        # 結果の構築
        result = {
            'expected_return': expected_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'risk_tolerance': risk_tolerance,
            'allocation': dict(zip(self.asset_names, optimal_weights)),
            'allocation_amounts': allocation_amounts,
            'investment_amount': investment_amount,
            'optimization_timestamp': datetime.now().isoformat(),
            'rebalancing_frequency': self.recommend_rebalancing_frequency(risk_tolerance)
        }
        
        print(f"✅ ポートフォリオ最適化完了 - 期待リターン: {expected_return:.2%}")
        return result
    
    def determine_risk_tolerance(self, risk_score, age):
        """
        リスク許容度の決定
        
        リスクスコアと年齢から、適切なリスク許容度を判定します。
        
        Args:
            risk_score (float): リスクスコア (0-100)
            age (int): 年齢
            
        Returns:
            str: リスク許容度 ('conservative', 'moderate', 'aggressive')
        """
        # 年齢による基本的なリスク許容度
        # 一般的に若いほどリスクを取れる
        age_factor = max(0, (70 - age) / 50)  # 70歳で0、20歳で1に近い値
        
        # リスクスコアによる調整
        risk_factor = max(0, (100 - risk_score) / 100)  # 高リスクスコアほど保守的に
        
        # 総合リスク許容度の計算
        total_risk_tolerance = (age_factor + risk_factor) / 2
        
        if total_risk_tolerance <= 0.3:
            return 'conservative'
        elif total_risk_tolerance <= 0.7:
            return 'moderate'
        else:
            return 'aggressive'
    
    def calculate_optimal_weights(self, risk_tolerance):
        """
        最適な資産配分ウェイトの計算
        
        リスク許容度に基づいて、効率的フロンティア上の最適点を見つけます。
        
        Args:
            risk_tolerance (str): リスク許容度
            
        Returns:
            numpy.ndarray: 最適ウェイト配列
        """
        # 期待リターンとボラティリティの配列を作成
        expected_returns = np.array([
            self.asset_classes[asset]['expected_return'] 
            for asset in self.asset_names
        ])
        
        volatilities = np.array([
            self.asset_classes[asset]['volatility'] 
            for asset in self.asset_names
        ])
        
        # 共分散行列の計算
        # Cov = D × Corr × D (Dは対角ボラティリティ行列)
        D = np.diag(volatilities)
        cov_matrix = D @ self.correlation_matrix @ D
        
        # 制約条件の設定
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 合計100%制約
        ]
        
        # 各資産の上下限制約
        bounds = []
        for asset in self.asset_names:
            min_weight = self.asset_classes[asset]['min_weight']
            max_weight = self.asset_classes[asset]['max_weight']
            bounds.append((min_weight, max_weight))
        
        # リスク許容度に応じた目的関数の設定
        if risk_tolerance == 'conservative':
            # リスク最小化（最小分散ポートフォリオ）
            def objective(weights):
                return weights.T @ cov_matrix @ weights
        elif risk_tolerance == 'moderate':
            # シャープレシオ最大化
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                return -(portfolio_return - 0.02) / portfolio_risk  # 負の値で最大化
        else:  # aggressive
            # リターン最大化（リスク制約下）
            def objective(weights):
                return -np.dot(weights, expected_returns)  # 負の値で最大化
            
            # アグレッシブな場合は追加のリスク制約
            max_risk = 0.20  # 最大20%のリスク
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: max_risk - np.sqrt(x.T @ cov_matrix @ x)
            })
        
        # 初期値の設定（等ウェイト）
        initial_weights = np.array([1.0 / len(self.asset_names)] * len(self.asset_names))
        
        # 最適化の実行
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x
        else:
            print("⚠️ 最適化に失敗しました。等ウェイトポートフォリオを返します。")
            return initial_weights
    
    def calculate_portfolio_return(self, weights):
        """
        ポートフォリオの期待リターン計算
        
        Args:
            weights (numpy.ndarray): 資産配分ウェイト
            
        Returns:
            float: ポートフォリオの期待年率リターン
        """
        expected_returns = np.array([
            self.asset_classes[asset]['expected_return'] 
            for asset in self.asset_names
        ])
        
        return np.dot(weights, expected_returns)
    
    def calculate_portfolio_risk(self, weights):
        """
        ポートフォリオのリスク（標準偏差）計算
        
        Args:
            weights (numpy.ndarray): 資産配分ウェイト
            
        Returns:
            float: ポートフォリオの年率標準偏差
        """
        volatilities = np.array([
            self.asset_classes[asset]['volatility'] 
            for asset in self.asset_names
        ])
        
        # 共分散行列の計算
        D = np.diag(volatilities)
        cov_matrix = D @ self.correlation_matrix @ D
        
        # ポートフォリオ分散の計算
        portfolio_variance = weights.T @ cov_matrix @ weights
        
        return np.sqrt(portfolio_variance)
    
    def recommend_rebalancing_frequency(self, risk_tolerance):
        """
        リバランス頻度の推奨
        
        Args:
            risk_tolerance (str): リスク許容度
            
        Returns:
            str: 推奨リバランス頻度
        """
        frequency_map = {
            'conservative': '年1回',
            'moderate': '半年に1回',
            'aggressive': '四半期に1回'
        }
        
        return frequency_map.get(risk_tolerance, '半年に1回')
    
    def generate_efficient_frontier(self, num_portfolios=100):
        """
        効率的フロンティアの生成
        
        リスク-リターン平面上の効率的フロンティアを計算します。
        
        Args:
            num_portfolios (int): 生成するポートフォリオ数
            
        Returns:
            dict: 効率的フロンティアのデータ
        """
        print(f"📊 効率的フロンティアを生成中（{num_portfolios}ポートフォリオ）...")
        
        expected_returns = np.array([
            self.asset_classes[asset]['expected_return'] 
            for asset in self.asset_names
        ])
        
        volatilities = np.array([
            self.asset_classes[asset]['volatility'] 
            for asset in self.asset_names
        ])
        
        # 共分散行列の計算
        D = np.diag(volatilities)
        cov_matrix = D @ self.correlation_matrix @ D
        
        # 最小リターンと最大リターンの設定
        min_return = min(expected_returns)
        max_return = max(expected_returns)
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # 制約条件の設定
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 合計100%制約
                {'type': 'eq', 'fun': lambda x, target=target_return: np.dot(x, expected_returns) - target}  # 目標リターン制約
            ]
            
            # 各資産の上下限制約
            bounds = []
            for asset in self.asset_names:
                min_weight = self.asset_classes[asset]['min_weight']
                max_weight = self.asset_classes[asset]['max_weight']
                bounds.append((min_weight, max_weight))
            
            # リスク最小化目的関数
            def objective(weights):
                return weights.T @ cov_matrix @ weights
            
            # 初期値
            initial_weights = np.array([1.0 / len(self.asset_names)] * len(self.asset_names))
            
            # 最適化実行
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                
                efficient_portfolios.append({
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'weights': weights,
                    'sharpe_ratio': (portfolio_return - 0.02) / portfolio_risk
                })
        
        print("✅ 効率的フロンティア生成完了")
        return {
            'portfolios': efficient_portfolios,
            'asset_names': self.asset_names,
            'generation_timestamp': datetime.now().isoformat()
        }

# テスト実行用の関数
def test_portfolio_optimizer():
    """
    ポートフォリオ最適化エンジンのテスト実行
    """
    print("🧪 ポートフォリオ最適化エンジンのテストを開始...")
    
    # エンジンの初期化
    optimizer = PortfolioOptimizer()
    
    # テスト用の顧客データ
    test_customers = [
        {
            'customer_id': 'TEST_001',
            'age': 30,
            'income': 6000000,
            'investment_amount': 2000000
        },
        {
            'customer_id': 'TEST_002',
            'age': 50,
            'income': 8000000,
            'investment_amount': 5000000
        },
        {
            'customer_id': 'TEST_003',
            'age': 65,
            'income': 4000000,
            'investment_amount': 10000000
        }
    ]
    
    # 各顧客のポートフォリオ最適化
    risk_scores = [25, 45, 65]  # 対応するリスクスコア
    
    for customer, risk_score in zip(test_customers, risk_scores):
        print(f"\n--- {customer['customer_id']} のポートフォリオ最適化 ---")
        print(f"年齢: {customer['age']}歳, リスクスコア: {risk_score}")
        
        result = optimizer.optimize_portfolio(customer, risk_score)
        
        print(f"期待リターン: {result['expected_return']:.2%}")
        print(f"リスク: {result['risk']:.2%}")
        print(f"シャープレシオ: {result['sharpe_ratio']:.3f}")
        print(f"リスク許容度: {result['risk_tolerance']}")
        print("資産配分:")
        for asset, weight in result['allocation'].items():
            if weight > 0.01:  # 1%以上の配分のみ表示
                amount = result['allocation_amounts'][asset]
                print(f"  {asset}: {weight:.1%} ({amount:,.0f}円)")
    
    print("\n✅ ポートフォリオ最適化エンジンのテスト完了")

if __name__ == "__main__":
    test_portfolio_optimizer()
