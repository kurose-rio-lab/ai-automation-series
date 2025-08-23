# AI小売システム セットアップガイド

## 🚀 クイックスタート（Windows）

### Step 1: Pythonのインストール
1. [Python公式サイト](https://www.python.org/downloads/) にアクセス
2. Python 3.9以上をダウンロード
3. インストール時に「Add Python to PATH」にチェック ✅

### Step 2: プロジェクトのダウンロード
1. このページの「Code」ボタンをクリック
2. 「Download ZIP」を選択
3. ダウンロードしたファイルを解凍

### Step 3: 必要ライブラリのインストール
```bash
# コマンドプロンプトを開いて以下を実行
cd ai-retail-systems
pip install -r requirements.txt

### Step 4: 動作確認
python inventory_optimization/basic_demo.py

## macOS/Linux版セットアップ
#Python3とpipがインストール済みの場合
git clone <repository-url>
cd ai-retail-systems
pip3 install -r requirements.txt
python3 inventory_optimization/basic_demo.py

## ❓ よくある問題と解決方法
Q: ModuleNotFoundError が出る
A: 以下のコマンドでライブラリを再インストール

pip install --upgrade pip
pip install -r requirements.txt

Q: Pythonが認識されない
A: 環境変数PATHにPythonを追加

システム環境変数を開く
PATH変数にPythonインストール先を追加
