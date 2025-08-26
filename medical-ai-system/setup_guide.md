# 医療AI統合診断支援システム セットアップガイド

このガイドでは、システムの詳細なセットアップ手順を説明します。

## 🖥️ システム要件

### 最小要件
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.9以上
- **RAM**: 4GB以上
- **ストレージ**: 2GB以上の空き容量
- **ネットワーク**: インターネット接続（初回セットアップ時）

### 推奨要件
- **OS**: Windows 11, macOS 12+, Ubuntu 20.04+
- **Python**: 3.11以上
- **RAM**: 8GB以上
- **ストレージ**: 10GB以上の空き容量（ログやデータ保存用）
- **CPU**: 4コア以上

## 📥 インストール手順

### Step 1: Pythonのインストール

#### Windows
1. [Python公式サイト](https://www.python.org/downloads/)からPython 3.11をダウンロード
2. インストーラを実行し、**「Add Python to PATH」にチェック**を入れる
3. 「Install Now」をクリック

#### macOS
```bash
# Homebrewを使用する場合
brew install python@3.11

# または公式インストーラを使用
# https://www.python.org/downloads/macos/
