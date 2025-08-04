#!/bin/bash

# =============================================================================
# AI自動レポート生成システム - セットアップスクリプト
# 黒瀬理央のAI研究室 - AIシステム開発シリーズ
# =============================================================================

set -e  # エラー時に即座に終了

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}=== $1 ===${NC}"
}

# システム情報取得
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# 必要コマンドの確認
check_command() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 is installed"
        return 0
    else
        log_error "$1 is not installed"
        return 1
    fi
}

# Python環境セットアップ
setup_python_env() {
    log_header "Python環境セットアップ"
    
    # Python 3.8+ チェック
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_info "Python version: $PYTHON_VERSION"
        
        # バージョンチェック
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 8 ]; then
            log_error "Python 3.8+ が必要です。現在のバージョン: $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "Python3 がインストールされていません"
        exit 1
    fi
    
    # 仮想環境作成
    if [ ! -d "venv" ]; then
        log_info "Python仮想環境を作成中..."
        python3 -m venv venv
        log_success "仮想環境を作成しました"
    else
        log_info "仮想環境は既に存在します"
    fi
    
    # 仮想環境有効化
    log_info "仮想環境を有効化中..."
    source venv/bin/activate
    
    # pip更新
    log_info "pipを最新版に更新中..."
    pip install --upgrade pip
    
    # 依存関係インストール
    if [ -f "requirements.txt" ]; then
        log_info "Python依存関係をインストール中..."
        pip install -r requirements.txt
        log_success "Python依存関係のインストール完了"
    else
        log_warning "requirements.txt が見つかりません"
    fi
}

# Node.js環境セットアップ
setup_nodejs_env() {
    log_header "Node.js環境セットアップ"
    
    # Node.js チェック
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        log_info "Node.js version: $NODE_VERSION"
        
        # Node.js 18+ チェック
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1 | sed 's/v//')
        if [ "$NODE_MAJOR" -lt 18 ]; then
            log_error "Node.js 18+ が必要です。現在のバージョン: $NODE_VERSION"
            exit 1
        fi
    else
        log_error "Node.js がインストールされていません"
        log_info "Node.jsをインストールしてください: https://nodejs.org/"
        exit 1
    fi
    
    # npm依存関係インストール
    if [ -f "package.json" ]; then
        log_info "Node.js依存関係をインストール中..."
        npm install
        log_success "Node.js依存関係のインストール完了"
    else
        log_warning "package.json が見つかりません"
    fi
}

# ディレクトリ構造作成
create_directories() {
    log_header "ディレクトリ構造作成"
    
    DIRECTORIES=(
        "logs"
        "temp"
        "uploads"
        "reports/output"
        "backups"
        "cache"
        "config/environment"
        "data/raw"
        "data/processed"
        "webhooks/logs"
        "exports/canva"
        "exports/reports"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_success "作成: $dir/"
        else
            log_info "既存: $dir/"
        fi
    done
}

# 環境変数ファイル設定
setup_env_file() {
    log_header "環境変数ファイル設定"
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_success ".env ファイルを作成しました"
            log_warning "重要: .env ファイルを編集して、実際のAPIキーを設定してください"
            log_info "編集コマンド: nano .env または vim .env"
        else
            log_warning ".env.example が見つかりません。手動で .env を作成してください"
        fi
    else
        log_info ".env ファイルは既に存在します"
    fi
}

# Google Apps Script設定
setup_google_apps_script() {
    log_header "Google Apps Script設定"
    
    if command -v clasp &> /dev/null; then
        log_success "clasp (Google Apps Script CLI) がインストール済み"
        
        if [ ! -f ".clasp.json" ]; then
            log_info "Google Apps Scriptプロジェクトを設定してください:"
            log_info "1. clasp login でログイン"
            log_info "2. clasp create --type standalone でプロジェクト作成"
            log_info "3. src/automation/data_cleaning_script.gs をアップロード"
        else
            log_info "Google Apps Script設定ファイルが存在します"
        fi
    else
        log_warning "clasp がインストールされていません"
        log_info "インストール: npm install -g @google/clasp"
    fi
}

# データベース設定チェック
check_database_setup() {
    log_header "データベース設定チェック"
    
    # MongoDB
    if command -v mongosh &> /dev/null || command -v mongo &> /dev/null; then
        log_success "MongoDB CLI が利用可能"
    else
        log_warning "MongoDB がインストールされていません（オプション）"
    fi
    
    # Redis
    if command -v redis-cli &> /dev/null; then
        log_success "Redis CLI が利用可能"
    else
        log_warning "Redis がインストールされていません（オプション）"
    fi
    
    # PostgreSQL
    if command -v psql &> /dev/null; then
        log_success "PostgreSQL CLI が利用可能"
    else
        log_warning "PostgreSQL がインストールされていません（オプション）"
    fi
}

# 権限設定
set_permissions() {
    log_header "ファイル権限設定"
    
    # スクリプトファイルに実行権限付与
    chmod +x scripts/*.sh 2>/dev/null || true
    
    # ログディレクトリ権限設定
    chmod 755 logs/ 2>/dev/null || true
    chmod 755 temp/ 2>/dev/null || true
    chmod 755 uploads/ 2>/dev/null || true
    
    log_success "ファイル権限を設定しました"
}

# 設定ファイル検証
validate_config() {
    log_header "設定ファイル検証"
    
    # 必須設定ファイルチェック
    REQUIRED_FILES=(
        ".env"
        "config/output_templates.json"
        "config/cost_optimization_config.json"
    )
    
    MISSING_FILES=0
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "必須ファイルが見つかりません: $file"
            MISSING_FILES=$((MISSING_FILES + 1))
        else
            log_success "設定ファイル確認: $file"
        fi
    done
    
    if [ $MISSING_FILES -gt 0 ]; then
        log_warning "$MISSING_FILES 個の必須ファイルが不足しています"
    fi
}

# テスト実行
run_tests() {
    log_header "システムテスト実行"
    
    # Python環境でのテスト
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        if [ -f "tests/test_basic.py" ]; then
            log_info "基本機能テストを実行中..."
            python -m pytest tests/test_basic.py -v || log_warning "一部のテストが失敗しました"
        fi
    fi
    
    # Node.js環境でのテスト
    if [ -f "package.json" ] && command -v npm &> /dev/null; then
        if npm run test > /dev/null 2>&1; then
            log_success "Node.jsテストが成功しました"
        else
            log_warning "Node.jsテストが失敗しました（正常な場合があります）"
        fi
    fi
}

# セットアップ完了メッセージ
show_completion_message() {
    log_header "セットアップ完了"
    
    echo ""
    log_success "🎉 AI自動レポート生成システムのセットアップが完了しました！"
    echo ""
    log_info "次のステップ:"
    echo "  1. .env ファイルを編集してAPIキーを設定"
    echo "  2. Google Apps Scriptプロジェクトを作成・設定"
    echo "  3. Zapier/Make.comでワークフローを設定"
    echo "  4. Canva APIアクセスを設定"
    echo ""
    log_info "システム起動コマンド:"
    echo "  • Python環境: source venv/bin/activate && python src/generators/report_generator_master.py"
    echo "  • Node.js環境: npm start"
    echo "  • 開発モード: npm run dev"
    echo ""
    log_info "サポート:"
    echo "  • YouTube: 黒瀬理央のAI研究室"
    echo "  • シリーズ: AIシステム開発シリーズ - ノーコード×AI完全習得"
    echo "  • ドキュメント: docs/implementation_guide.md"
    echo ""
    log_warning "重要: セキュリティのため、APIキーを適切に設定し、.envファイルを保護してください"
}

# メイン実行関数
main() {
    log_header "AI自動レポート生成システム セットアップ開始"
    
    # システム情報表示
    OS_TYPE=$(detect_os)
    log_info "検出されたOS: $OS_TYPE"
    log_info "作業ディレクトリ: $(pwd)"
    
    # 必要コマンドチェック
    log_header "必要コマンドチェック"
    REQUIRED_COMMANDS=("python3" "node" "npm" "git")
    MISSING_COMMANDS=0
    
    for cmd in "${REQUIRED_COMMANDS[@]}"; do
        if ! check_command "$cmd"; then
            MISSING_COMMANDS=$((MISSING_COMMANDS + 1))
        fi
    done
    
    if [ $MISSING_COMMANDS -gt 0 ]; then
        log_error "$MISSING_COMMANDS 個の必要なコマンドが見つかりません"
        log_info "不足しているコマンドをインストールしてから再実行してください"
        exit 1
    fi
    
    # セットアップ実行
    setup_python_env
    setup_nodejs_env
    create_directories
    setup_env_file
    setup_google_apps_script
    check_database_setup
    set_permissions
    validate_config
    run_tests
    show_completion_message
    
    log_success "✅ セットアップが正常に完了しました！"
}

# スクリプト実行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
