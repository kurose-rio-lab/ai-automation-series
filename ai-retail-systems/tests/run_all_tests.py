"""
全テストの実行スクリプト
"""

import unittest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_all_tests():
    """すべてのテストを実行"""
    
    # テストスイートの作成
    test_suite = unittest.TestSuite()
    
    # 各テストモジュールを追加
    test_modules = [
        'test_inventory_optimization',
        'test_demand_forecasting', 
        'test_integration'
    ]
    
    for module_name in test_modules:
        try:
            # モジュールの動的インポート
            module = __import__(module_name)
            
            # テストケースをスイートに追加
            test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(module))
            
            print(f"✓ {module_name} を追加しました")
            
        except ImportError as e:
            print(f"✗ {module_name} のインポートに失敗: {e}")
    
    # テストランナーの設定
    runner = unittest.TextTestRunner(
        verbosity=2,  # 詳細な出力
        stream=sys.stdout,
        descriptions=True,
        failfast=False  # 最初の失敗で停止しない
    )
    
    print("\n" + "="*60)
    print("AI小売システム - 全テスト実行開始")
    print("="*60)
    
    # テスト実行
    result = runner.run(test_suite)
    
    print("\n" + "="*60)
    print("テスト実行結果サマリー")
    print("="*60)
    print(f"実行テスト数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    
    if result.failures:
        print("\n失敗したテスト:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nエラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # 総合結果
    if result.wasSuccessful():
        print(f"\n🎉 すべてのテストが成功しました！")
        return True
    else:
        print(f"\n❌ 一部のテストが失敗しました。")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
