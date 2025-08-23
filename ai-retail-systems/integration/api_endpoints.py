"""
AI小売システム APIエンドポイント
Webアプリケーション用のREST API
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime
from system_integration import AIRetailSystemIntegrator

app = Flask(__name__)
CORS(app)  # CORS対応

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# システム統合インスタンス
integrator = AIRetailSystemIntegrator()

@app.route('/api/health', methods=['GET'])
def health_check():
    """ヘルスチェックエンドポイント"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/products/<product_id>/analysis', methods=['GET'])
def get_product_analysis(product_id):
    """製品分析API"""
    try:
        result = integrator.comprehensive_analysis(product_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"製品分析エラー: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/products/batch-analysis', methods=['POST'])
def batch_product_analysis():
    """一括製品分析API"""
    try:
        data = request.get_json()
        product_ids = data.get('product_ids', [])
        
        if not product_ids:
            return jsonify({'error': 'product_idsが必要です'}), 400
        
        results = integrator.batch_analysis(product_ids)
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"一括分析エラー: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/summary', methods=['GET'])
def get_dashboard_summary():
    """ダッシュボードサマリーAPI"""
    try:
        summary = integrator.get_dashboard_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"ダッシュボードエラー: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory/alerts', methods=['GET'])
def get_inventory_alerts():
    """在庫アラートAPI"""
    try:
        # 全製品の在庫状況をチェック
        import pandas as pd
        
        inventory_df = pd.read_csv('data/sample_data/inventory.csv')
        alerts = []
        
        for _, row in inventory_df.iterrows():
            if row['current_stock'] <= row['min_stock']:
                alerts.append({
                    'product_id': row['product_id'],
                    'current_stock': row['current_stock'],
                    'min_stock': row['min_stock'],
                    'alert_type': 'critical_low',
                    'recommended_action': 'immediate_reorder'
                })
            elif row['current_stock'] <= row['min_stock'] * 1.5:
                alerts.append({
                    'product_id': row['product_id'],
                    'current_stock': row['current_stock'],
                    'min_stock': row['min_stock'],
                    'alert_type': 'low',
                    'recommended_action': 'schedule_reorder'
                })
        
        return jsonify({
            'alerts': alerts,
            'alert_count': len(alerts),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"在庫アラートエラー: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
