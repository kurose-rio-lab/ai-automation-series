# スクリプト説明

## 🔧 setup.sh
自動環境構築スクリプト

### 実行方法
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh

実行内容
Python/Node.js環境チェック
仮想環境作成・依存関係インストール
ディレクトリ構造作成
環境変数ファイル生成
基本テスト実行
🚢 deploy.sh
デプロイ自動化スクリプト

実行方法
chmod +x scripts/deploy.sh
./scripts/deploy.sh [environment] [options]

オプション
development: 開発環境
staging: ステージング環境
production: 本番環境
--rollback: ロールバック実行
安全機能
自動バックアップ
ヘルスチェック
段階的デプロイ
失敗時自動ロールバック
