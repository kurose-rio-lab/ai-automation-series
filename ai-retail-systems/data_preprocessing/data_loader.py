"""
データローダー・前処理システム
各種データソースからの統合的なデータ読み込みと標準化
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime, timedelta
import re
import json
from typing import Dict, List, Optional, Union

class DataLoader:
    def __init__(self, config_path="config/config.yaml"):
        """データローダーの初期化"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # ログ設定
        logging.basicConfig(
            level=getattr(logging, self.config['monitoring']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # データスキーマ定義
        self.schemas = {
            'sales': {
                'required_columns': ['date', 'product_id', 'quantity', 'price'],
                'date_columns': ['date'],
                'numeric_columns': ['quantity', 'price', 'sales_amount']
            },
            'inventory': {
                'required_columns': ['product_id', 'current_stock'],
                'numeric_columns': ['current_stock', 'reorder_point', 'safety_stock']
            },
            'customers': {
                'required_columns': ['customer_id'],
                'date_columns': ['registration_date', 'last_purchase_date'],
                'numeric_columns': ['total_spent', 'order_count']
            },
            'products': {
                'required_columns': ['product_id', 'product_name'],
                'numeric_columns': ['price', 'cost', 'weight']
            }
        }
    
    def validate_data_schema(self, df: pd.DataFrame, schema_name: str) -> bool:
        """データスキーマの検証"""
        if schema_name not in self.schemas:
            self.logger.warning(f"未知のスキーマ: {schema_name}")
            return True
        
        schema = self.schemas[schema_name]
        
        # 必須列の確認
        required_columns = schema.get('required_columns', [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"必須列が不足: {missing_columns}")
            return False
        
        # データ型の確認
        numeric_columns = schema.get('numeric_columns', [])
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                self.logger.warning(f"数値列でない可能性: {col}")
        
        self.logger.info(f"スキーマ検証完了: {schema_name}")
        return True
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """列名のクリーニング"""
        # 空白の除去、小文字化、特殊文字の置換
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('[^\w]', '_', regex=True)
        df.columns = df.columns.str.lower()
        
        return df
    
    def standardize_date_columns(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """日付列の標準化"""
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    self.logger.info(f"日付変換完了: {col}")
                except Exception as e:
                    self.logger.error(f"日付変換エラー ({col}): {e}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """欠損値処理"""
        initial_nulls = df.isnull().sum().sum()
        
        if strategy == 'auto':
            for column in df.columns:
                null_ratio = df[column].isnull().sum() / len(df)
                
                if null_ratio > 0.5:
                    # 50%以上欠損している列は削除を検討
                    self.logger.warning(f"列 {column} の欠損率が高い: {null_ratio:.2%}")
                    continue
                
                if pd.api.types.is_numeric_dtype(df[column]):
                    # 数値列は中央値で埋める
                    df[column].fillna(df[column].median(), inplace=True)
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    # 日付列は前方埋め
                    df[column].fillna(method='ffill', inplace=True)
                else:
                    # 文字列列は最頻値で埋める
                    mode_value = df[column].mode()
                    if len(mode_value) > 0:
                        df[column].fillna(mode_value[0], inplace=True)
                    else:
                        df[column].fillna('Unknown', inplace=True)
        
        final_nulls = df.isnull().sum().sum()
        self.logger.info(f"欠損値処理完了: {initial_nulls} → {final_nulls}")
        
        return df
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, numeric_columns: List[str] = None, method: str = 'iqr') -> pd.DataFrame:
        """外れ値検出・処理"""
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_summary = {}
        
        for column in numeric_columns:
            if column not in df.columns:
                continue
            
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                outlier_count = len(outliers)
                
                # 外れ値を境界値にクリップ
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                
                outlier_summary[column] = {
                    'count': outlier_count,
                    'percentage': outlier_count / len(df) * 100,
                    'bounds': (lower_bound, upper_bound)
                }
        
        self.logger.info(f"外れ値処理完了: {len(outlier_summary)}列処理")
        
        return df, outlier_summary
    
    def load_sales_data(self, file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
        """売上データの読み込み"""
        try:
            df = pd.read_csv(file_path, **kwargs)
            self.logger.info(f"売上データ読み込み: {len(df)}レコード")
            
            # 列名クリーニング
            df = self.clean_column_names(df)
            
            # スキーマ検証
            if not self.validate_data_schema(df, 'sales'):
                return None
            
            # 日付列の標準化
            df = self.standardize_date_columns(df, ['date'])
            
            # 売上金額の計算（存在しない場合）
            if 'sales_amount' not in df.columns and all(col in df.columns for col in ['quantity', 'price']):
                df['sales_amount'] = df['quantity'] * df['price']
                self.logger.info("売上金額を計算して追加")
            
            # 欠損値処理
            df = self.handle_missing_values(df)
            
            # 外れ値処理
            df, outlier_info = self.detect_and_handle_outliers(df, ['quantity', 'price', 'sales_amount'])
            
            # データ品質チェック
            quality_report = self._generate_data_quality_report(df, 'sales')
            
            return df
            
        except Exception as e:
            self.logger.error(f"売上データ読み込みエラー: {e}")
            return None
    
    def load_inventory_data(self, file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
        """在庫データの読み込み"""
        try:
            df = pd.read_csv(file_path, **kwargs)
            self.logger.info(f"在庫データ読み込み: {len(df)}レコード")
            
            # 列名クリーニング
            df = self.clean_column_names(df)
            
            # スキーマ検証
            if not self.validate_data_schema(df, 'inventory'):
                return None
            
            # 欠損値処理
            df = self.handle_missing_values(df)
            
            # 在庫関連の計算列追加
            if all(col in df.columns for col in ['current_stock', 'reorder_point']):
                df['days_until_reorder'] = np.where(
                    df['current_stock'] <= df['reorder_point'], 0,
                    (df['current_stock'] - df['reorder_point']) / df.get('avg_daily_demand', 1)
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"在庫データ読み込みエラー: {e}")
            return None
    
    def load_customer_data(self, file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
        """顧客データの読み込み"""
        try:
            df = pd.read_csv(file_path, **kwargs)
            self.logger.info(f"顧客データ読み込み: {len(df)}レコード")
            
            # 列名クリーニング
            df = self.clean_column_names(df)
            
            # スキーマ検証
            if not self.validate_data_schema(df, 'customers'):
                return None
            
            # 日付列の標準化
            date_columns = ['registration_date', 'last_purchase_date', 'birth_date']
            df = self.standardize_date_columns(df, date_columns)
            
            # 顧客セグメント計算
            if 'total_spent' in df.columns:
                df['customer_segment'] = pd.cut(
                    df['total_spent'],
                    bins=[-float('inf'), 10000, 50000, 100000, float('inf')],
                    labels=['Bronze', 'Silver', 'Gold', 'Platinum']
                )
            
            # 年齢計算（生年月日がある場合）
            if 'birth_date' in df.columns:
                df['age'] = (datetime.now() - df['birth_date']).dt.days / 365.25
                df['age'] = df['age'].round().astype('Int64')
            
            return df
            
        except Exception as e:
            self.logger.error(f"顧客データ読み込みエラー: {e}")
            return None
    
    def load_product_data(self, file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
        """商品データの読み込み"""
        try:
            df = pd.read_csv(file_path, **kwargs)
            self.logger.info(f"商品データ読み込み: {len(df)}レコード")
            
            # 列名クリーニング
            df = self.clean_column_names(df)
            
            # スキーマ検証
            if not self.validate_data_schema(df, 'products'):
                return None
            
            # 利益率計算（価格とコストがある場合）
            if all(col in df.columns for col in ['price', 'cost']):
                df['profit_margin'] = (df['price'] - df['cost']) / df['price'] * 100
                df['profit_amount'] = df['price'] - df['cost']
            
            # カテゴリの標準化
            if 'category' in df.columns:
                df['category'] = df['category'].str.strip().str.title()
            
            return df
            
        except Exception as e:
            self.logger.error(f"商品データ読み込みエラー: {e}")
            return None
    
    def load_multiple_files(self, file_patterns: Dict[str, str], data_dir: Union[str, Path] = "data") -> Dict[str, pd.DataFrame]:
        """複数ファイルの一括読み込み"""
        data_dir = Path(data_dir)
        loaded_data = {}
        
        for data_type, pattern in file_patterns.items():
            matching_files = list(data_dir.glob(pattern))
            
            if not matching_files:
                self.logger.warning(f"パターン '{pattern}' にマッチするファイルが見つかりません")
                continue
            
            # 複数ファイルがある場合は結合
            if len(matching_files) > 1:
                dfs = []
                for file_path in matching_files:
                    df = self._load_by_data_type(file_path, data_type)
                    if df is not None:
                        df['source_file'] = file_path.name
                        dfs.append(df)
                
                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    loaded_data[data_type] = combined_df
                    self.logger.info(f"{data_type}データ結合完了: {len(combined_df)}レコード")
            else:
                # 単一ファイル
                df = self._load_by_data_type(matching_files[0], data_type)
                if df is not None:
                    loaded_data[data_type] = df
        
        return loaded_data
    
    def _load_by_data_type(self, file_path: Path, data_type: str) -> Optional[pd.DataFrame]:
        """データタイプに応じた読み込み"""
        if data_type == 'sales':
            return self.load_sales_data(file_path)
        elif data_type == 'inventory':
            return self.load_inventory_data(file_path)
        elif data_type == 'customers':
            return self.load_customer_data(file_path)
        elif data_type == 'products':
            return self.load_product_data(file_path)
        else:
            # 汎用CSV読み込み
            try:
                df = pd.read_csv(file_path)
                df = self.clean_column_names(df)
                return df
            except Exception as e:
                self.logger.error(f"ファイル読み込みエラー ({file_path}): {e}")
                return None
    
    def _generate_data_quality_report(self, df: pd.DataFrame, data_type: str) -> Dict:
        """データ品質レポート生成"""
        report = {
            'data_type': data_type,
            'record_count': len(df),
            'column_count': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # 数値列の統計
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            report['numeric_stats'] = df[numeric_columns].describe().to_dict()
        
        # 文字列列のユニーク値数
        string_columns = df.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            report['categorical_stats'] = {col: df[col].nunique() for col in string_columns}
        
        return report
    
    def export_processed_data(self, data_dict: Dict[str, pd.DataFrame], output_dir: Union[str, Path] = "data/processed") -> Dict[str, Path]:
        """処理済みデータのエクスポート"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for data_type, df in data_dict.items():
            # ファイル名生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_processed_{timestamp}.csv"
            file_path = output_dir / filename
            
            try:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                saved_files[data_type] = file_path
                self.logger.info(f"{data_type}データ保存完了: {file_path}")
            except Exception as e:
                self.logger.error(f"データ保存エラー ({data_type}): {e}")
        
        return saved_files
    
    def create_data_summary_report(self, data_dict: Dict[str, pd.DataFrame]) -> str:
        """データサマリーレポート作成"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("データローダー処理レポート")
        report_lines.append("=" * 60)
        report_lines.append(f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"処理データ種別数: {len(data_dict)}")
        
        total_records = sum(len(df) for df in data_dict.values())
        report_lines.append(f"総レコード数: {total_records:,}")
        
        report_lines.append("\n【データ種別別サマリー】")
        for data_type, df in data_dict.items():
            report_lines.append(f"\n■ {data_type.upper()}")
            report_lines.append(f"  レコード数: {len(df):,}")
            report_lines.append(f"  列数: {len(df.columns)}")
            
            # 欠損値情報
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                report_lines.append(f"  欠損値: {missing_count:,} ({missing_count/df.size*100:.2f}%)")
            
            # 重複レコード
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                report_lines.append(f"  重複レコード: {duplicate_count:,}")
            
            # 日付範囲（日付列がある場合）
            date_columns = df.select_dtypes(include=['datetime64']).columns
            for date_col in date_columns:
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                report_lines.append(f"  {date_col}範囲: {min_date.date()} ～ {max_date.date()}")
        
        report_text = "\n".join(report_lines)
        
        # レポート保存
        output_dir = Path("results/data_loading")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"data_loading_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        self.logger.info(f"データサマリーレポート保存: {report_file}")
        
        return report_text

def create_sample_datasets():
    """サンプルデータセットの生成"""
    np.random.seed(42)
    output_dir = Path("data/sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 商品マスタデータ
    products_data = []
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports']
    
    for i in range(1, 101):
        products_data.append({
            'product_id': f'PROD_{i:04d}',
            'product_name': f'Product {i}',
            'category': np.random.choice(categories),
            'price': np.random.uniform(1000, 10000),
            'cost': np.random.uniform(300, 5000),
            'weight': np.random.uniform(0.1, 5.0),
            'supplier_id': f'SUP_{np.random.randint(1, 21):02d}'
        })
    
    products_df = pd.DataFrame(products_data)
    products_df.to_csv(output_dir / "products_master.csv", index=False, encoding='utf-8-sig')
    
    # 2. 顧客マスタデータ
    customers_data = []
    
    for i in range(1, 501):
        reg_date = datetime.now() - timedelta(days=np.random.randint(30, 1095))
        last_purchase = reg_date + timedelta(days=np.random.randint(0, 365))
        
        customers_data.append({
            'customer_id': f'CUST_{i:04d}',
            'customer_name': f'Customer {i}',
            'email': f'customer{i}@example.com',
            'registration_date': reg_date,
            'last_purchase_date': last_purchase,
            'total_spent': np.random.exponential(scale=50000),
            'order_count': np.random.poisson(lam=5),
            'birth_date': datetime.now() - timedelta(days=np.random.randint(18*365, 70*365))
        })
    
    customers_df = pd.DataFrame(customers_data)
    customers_df.to_csv(output_dir / "customers_master.csv", index=False, encoding='utf-8-sig')
    
    # 3. 詳細売上データ（既存のものを拡張）
    if not (output_dir / "detailed_sales_data.csv").exists():
        sales_data = []
        
        for _ in range(10000):
            date = datetime.now() - timedelta(days=np.random.randint(0, 365))
            product_id = f'PROD_{np.random.randint(1, 101):04d}'
            customer_id = f'CUST_{np.random.randint(1, 501):04d}'
            quantity = np.random.poisson(lam=2) + 1
            
            # 商品価格を取得（簡略化のため固定範囲）
            base_price = np.random.uniform(1000, 10000)
            discount = np.random.uniform(0.9, 1.0)
            price = base_price * discount
            
            sales_data.append({
                'date': date,
                'product_id': product_id,
                'customer_id': customer_id,
                'quantity': quantity,
                'price': round(price, 2),
                'sales_amount': round(quantity * price, 2),
                'channel': np.random.choice(['Online', 'Store', 'Mobile']),
                'payment_method': np.random.choice(['Credit', 'Debit', 'Cash', 'PayPal'])
            })
        
        sales_df = pd.DataFrame(sales_data)
        sales_df.to_csv(output_dir / "detailed_sales_data.csv", index=False, encoding='utf-8-sig')
    
    print(f"サンプルデータセット生成完了: {output_dir}")

def main():
    """メイン実行関数"""
    print("データローダー・前処理システムを開始します...")
    
    # サンプルデータ生成
    if not Path("data/sample_data/products_master.csv").exists():
        print("サンプルデータセットを生成中...")
        create_sample_datasets()
    
    # データローダー初期化
    loader = DataLoader()
    
    # ファイルパターン定義
    file_patterns = {
        'sales': '*sales*.csv',
        'products': '*products*.csv',
        'customers': '*customers*.csv',
        'inventory': '*inventory*.csv'
    }
    
    # 複数ファイル読み込み
    print("データファイルを読み込み中...")
    loaded_data = loader.load_multiple_files(file_patterns, "data/sample_data")
    
    if not loaded_data:
        print("読み込み可能なデータファイルが見つかりませんでした")
        return
    
    # データ品質チェック・サマリー出力
    summary_report = loader.create_data_summary_report(loaded_data)
    print("\n" + summary_report)
    
    # 処理済みデータのエクスポート
    saved_files = loader.export_processed_data(loaded_data)
    
    print(f"\n✅ データローダー処理が完了しました！")
    print(f"処理データ種別数: {len(loaded_data)}")
    print(f"保存ファイル数: {len(saved_files)}")
    
    for data_type, file_path in saved_files.items():
        print(f"  {data_type}: {file_path}")

if __name__ == "__main__":
    main()
