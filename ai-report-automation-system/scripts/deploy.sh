#!/bin/bash

# =============================================================================
# AI自動レポート生成システム - デプロイスクリプト
# 黒瀬理央のAI研究室 - AIシステム開発シリーズ
# =============================================================================

set -e  # エラー時に即座に終了

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 設定
DEPLOY_ENV=${1:-production}
BACKUP_RETAIN_DAYS=30
HEALTH_CHECK_TIMEOUT=60
MAX_DEPLOY_ATTEMPTS=3

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

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# 設定読み込み
load_deploy_config() {
    log_header "デプロイ設定読み込み"
    
    case $DEPLOY_ENV in
        "development")
            ENV_FILE=".env.development"
            DEPLOY_TARGET="dev"
            ;;
        "staging")
            ENV_FILE=".env.staging"
            DEPLOY_TARGET="staging"
            ;;
        "production")
            ENV_FILE=".env.production"
            DEPLOY_TARGET="prod"
            ;;
        *)
            log_error "無効なデプロイ環境: $DEPLOY_ENV"
            log_info "有効な環境: development, staging, production"
            exit 1
            ;;
    esac
    
    log_info "デプロイ環境: $DEPLOY_ENV"
    log_info "設定ファイル: $ENV_FILE"
    
    if [ ! -f "$ENV_FILE" ]; then
        log_error "環境設定ファイルが見つかりません: $ENV_FILE"
        exit 1
    fi
    
    # 環境変数読み込み
    source "$ENV_FILE"
    log_success "環境設定を読み込みました"
}

# 事前チェック
pre_deploy_checks() {
    log_header "デプロイ前チェック"
    
    # Gitリポジトリチェック
    if [ ! -d ".git" ]; then
        log_error "Gitリポジトリではありません"
        exit 1
    fi
    
    # 未コミット変更チェック
    if ! git diff-index --quiet HEAD --; then
        log_warning "未コミットの変更があります"
        if [ "$DEPLOY_ENV" = "production" ]; then
            log_error "本番環境へのデプロイ前に全ての変更をコミットしてください"
            exit 1
        fi
    fi
    
    # ブランチチェック
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    case $DEPLOY_ENV in
        "production")
            if [ "$CURRENT_BRANCH" != "main" ] && [ "$CURRENT_BRANCH" != "master" ]; then
                log_error "本番環境デプロイは main/master ブランチから実行してください"
                log_info "現在のブランチ: $CURRENT_BRANCH"
                exit 1
            fi
            ;;
        "staging")
            if [ "$CURRENT_BRANCH" != "develop" ] && [ "$CURRENT_BRANCH" != "main" ]; then
                log_warning "ステージング環境は通常 develop ブランチからデプロイします"
                log_info "現在のブランチ: $CURRENT_BRANCH"
            fi
            ;;
    esac
    
    log_info "現在のブランチ: $CURRENT_BRANCH"
    
    # 必要ファイルチェック
    REQUIRED_FILES=(
        "src/generators/report_generator_master.py"
        "src/utils/chart_generator.js"
        "config/output_templates.json"
        "$ENV_FILE"
    )
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "必要ファイルが見つかりません: $file"
            exit 1
        fi
    done
    
    log_success "事前チェック完了"
}

# バックアップ作成
create_backup() {
    log_header "バックアップ作成"
    
    BACKUP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_DIR="backups/deploy_${DEPLOY_ENV}_${BACKUP_TIMESTAMP}"
    
    mkdir -p "$BACKUP_DIR"
    
    # 設定ファイルバックアップ
    cp -r config/ "$BACKUP_DIR/" 2>/dev/null || true
    cp .env* "$BACKUP_DIR/" 2>/dev/null || true
    
    # データベースバックアップ（MongoDB）
    if command -v mongodump &> /dev/null && [ -n "$MONGODB_URI" ]; then
        log_info "MongoDBバックアップ作成中..."
        mongodump --uri="$MONGODB_URI" --out "$BACKUP_DIR/mongodb" || log_warning "MongoDBバックアップに失敗"
    fi
    
    # 重要ファイルのバックアップ
    cp -r reports/output/ "$BACKUP_DIR/reports_output/" 2>/dev/null || true
    cp -r logs/ "$BACKUP_DIR/logs/" 2>/dev/null || true
    
    # バックアップ情報記録
    cat > "$BACKUP_DIR/backup_info.txt" << EOF
Backup created: $(date)
Environment: $DEPLOY_ENV
Git commit: $(git rev-parse HEAD)
Branch: $(git rev-parse --abbrev-ref HEAD)
User: $(whoami)
Host: $(hostname)
EOF
    
    log_success "バックアップ作成完了: $BACKUP_DIR"
    
    # 古いバックアップ削除
    find backups/ -name "deploy_${DEPLOY_ENV}_*" -type d -mtime +$BACKUP_RETAIN_DAYS -exec rm -rf {} + 2>/dev/null || true
    log_info "${BACKUP_RETAIN_DAYS}日以前のバックアップを削除しました"
}

# 依存関係更新
update_dependencies() {
    log_header "依存関係更新"
    
    # Python依存関係更新
    if [ -f "requirements.txt" ]; then
        log_step "Python依存関係を更新中..."
        source venv/bin/activate 2>/dev/null || {
            log_warning "Python仮想環境が見つかりません。作成中..."
            python3 -m venv venv
            source venv/bin/activate
        }
        
        pip install --upgrade pip
        pip install -r requirements.txt
        log_success "Python依存関係更新完了"
    fi
    
    # Node.js依存関係更新
    if [ -f "package.json" ]; then
        log_step "Node.js依存関係を更新中..."
        npm ci --production
        log_success "Node.js依存関係更新完了"
    fi
}

# ビルド実行
build_application() {
    log_header "アプリケーションビルド"
    
    # フロントエンド資産ビルド
    if [ -f "package.json" ] && npm run build &> /dev/null; then
        log_success "フロントエンド資産をビルドしました"
    fi
    
    # Python バイトコード最適化
    if [ -d "venv" ]; then
        source venv/bin/activate
        python -m compileall src/ -b -q || log_warning "Pythonバイトコード最適化に失敗"
    fi
    
    # 設定ファイル検証
    log_step "設定ファイル検証中..."
    python3 -c "
import json
import sys

try:
    with open('config/output_templates.json', 'r') as f:
        json.load(f)
    with open('config/cost_optimization_config.json', 'r') as f:
        json.load(f)
    print('設定ファイル検証成功')
except Exception as e:
    print(f'設定ファイル検証エラー: {e}')
    sys.exit(1)
" || exit 1
    
    log_success "アプリケーションビルド完了"
}

# テスト実行
run_deployment_tests() {
    log_header "デプロイメントテスト実行"
    
    # ユニットテスト
    if [ -f "requirements.txt" ] && [ -d "tests" ]; then
        log_step "ユニットテスト実行中..."
        source venv/bin/activate
        python -m pytest tests/ -x --tb=short || {
            log_error "ユニットテストが失敗しました"
            if [ "$DEPLOY_ENV" = "production" ]; then
                exit 1
            else
                log_warning "テスト失敗を無視して続行します（非本番環境）"
            fi
        }
        log_success "ユニットテスト完了"
    fi
    
    # Node.jsテスト
    if [ -f "package.json" ]; then
        log_step "Node.jsテスト実行中..."
        npm test || {
            log_warning "Node.jsテストが失敗しました"
            if [ "$DEPLOY_ENV" = "production" ]; then
                log_error "本番環境デプロイ時はテストが必須です"
                exit 1
            fi
        }
    fi
    
    # 統合テスト
    log_step "システム統合テスト実行中..."
    if [ -f "tests/integration_test.py" ]; then
        source venv/bin/activate
        python tests/integration_test.py || log_warning "統合テストが失敗しました"
    fi
    
    log_success "デプロイメントテスト完了"
}

# アプリケーション停止
stop_application() {
    log_header "アプリケーション停止"
    
    # PM2で管理されているプロセス停止
    if command -v pm2 &> /dev/null; then
        pm2 stop all || log_warning "PM2プロセス停止に失敗"
        pm2 delete all || log_warning "PM2プロセス削除に失敗"
    fi
    
    # PIDファイルベースの停止
    if [ -f "logs/app.pid" ]; then
        PID=$(cat logs/app.pid)
        if kill -0 "$PID" 2>/dev/null; then
            log_info "アプリケーションプロセスを停止中... (PID: $PID)"
            kill -TERM "$PID"
            sleep 5
            if kill -0 "$PID" 2>/dev/null; then
                kill -KILL "$PID"
                log_warning "強制終了しました"
            fi
        fi
        rm -f logs/app.pid
    fi
    
    # ポート使用プロセス停止
    if [ -n "$PORT" ]; then
        PROCESS_ON_PORT=$(lsof -ti:$PORT 2>/dev/null || true)
        if [ -n "$PROCESS_ON_PORT" ]; then
            log_info "ポート $PORT を使用中のプロセスを停止中..."
            kill -TERM $PROCESS_ON_PORT || true
        fi
    fi
    
    log_success "アプリケーション停止完了"
}

# アプリケーション起動
start_application() {
    log_header "アプリケーション起動"
    
    # 環境変数設定
    export NODE_ENV=$DEPLOY_ENV
    export DEPLOY_TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    
    case $DEPLOY_ENV in
        "production")
            # 本番環境：PM2で起動
            if command -v pm2 &> /dev/null; then
                log_step "PM2でアプリケーションを起動中..."
                pm2 start ecosystem.config.js --env production || {
                    log_warning "PM2設定ファイルが見つかりません。直接起動します"
                    pm2 start "npm run start" --name "ai-report-system"
                }
            else
                log_step "バックグラウンドでアプリケーションを起動中..."
                nohup npm start > logs/app.log 2>&1 &
                echo $! > logs/app.pid
            fi
            ;;
        "staging"|"development")
            # 開発/ステージング環境：直接起動
            log_step "開発モードでアプリケーションを起動中..."
            npm run dev > logs/app.log 2>&1 &
            echo $! > logs/app.pid
            ;;
    esac
    
    log_success "アプリケーション起動完了"
}

# ヘルスチェック
health_check() {
    log_header "ヘルスチェック実行"
    
    local port=${PORT:-3000}
    local max_attempts=$((HEALTH_CHECK_TIMEOUT / 5))
    local attempt=1
    
    log_info "ヘルスチェック開始 (ポート: $port, タイムアウト: ${HEALTH_CHECK_TIMEOUT}秒)"
    
    while [ $attempt -le $max_attempts ]; do
        log_step "ヘルスチェック試行 $attempt/$max_attempts"
        
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            log_success "✅ アプリケーションが正常に起動しました"
            return 0
        fi
        
        # 基本的な接続チェック
        if nc -z localhost $port 2>/dev/null; then
            log_success "ポート接続確認済み。アプリケーション準備中..."
            sleep 5
            continue
        fi
        
        log_warning "接続試行中... ($attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done
    
    log_error "❌ ヘルスチェックがタイムアウトしました"
    
    # デバッグ情報出力
    log_info "デバッグ情報:"
    log_info "プロセス状況:"
    ps aux | grep -E "(node|python)" | head -5
    log_info "ポート使用状況:"
    netstat -tuln | grep ":$port " || echo "ポート $port は使用されていません"
    log_info "最新ログ:"
    tail -10 logs/app.log 2>/dev/null || echo "ログファイルが見つかりません"
    
    return 1
}

# デプロイ後処理
post_deploy_tasks() {
    log_header "デプロイ後処理"
    
    # キャッシュクリア
    log_step "キャッシュクリア中..."
    rm -rf cache/* 2>/dev/null || true
    rm -rf temp/* 2>/dev/null || true
    
    # ログローテーション
    log_step "ログローテーション実行中..."
    if [ -f "logs/app.log" ] && [ $(stat -f%z "logs/app.log" 2>/dev/null || stat -c%s "logs/app.log" 2>/dev/null || echo 0) -gt 10485760 ]; then
        mv logs/app.log "logs/app_$(date +%Y%m%d_%H%M%S).log"
        touch logs/app.log
    fi
    
    # 権限設定
    log_step "権限設定中..."
    chmod 755 scripts/*.sh 2>/dev/null || true
    chmod 644 config/*.json 2>/dev/null || true
    chmod 600 .env* 2>/dev/null || true
    
    # デプロイ情報記録
    cat > "logs/deploy_${DEPLOY_ENV}_$(date +%Y%m%d_%H%M%S).log" << EOF
=== デプロイ情報 ===
日時: $(date)
環境: $DEPLOY_ENV
Gitコミット: $(git rev-parse HEAD)
ブランチ: $(git rev-parse --abbrev-ref HEAD)
ユーザー: $(whoami)
ホスト: $(hostname)
バージョン: $(git describe --tags --always 2>/dev/null || echo "unknown")

=== システム情報 ===
OS: $(uname -a)
Python: $(python3 --version 2>/dev/null || echo "N/A")
Node.js: $(node --version 2>/dev/null || echo "N/A")
npm: $(npm --version 2>/dev/null || echo "N/A")

=== 環境変数（セキュア）===
NODE_ENV: $NODE_ENV
PORT: ${PORT:-"未設定"}
DEPLOY_TARGET: $DEPLOY_TARGET

デプロイ成功: $(date)
EOF
    
    log_success "デプロイ後処理完了"
}

# ロールバック機能
rollback() {
    log_header "ロールバック実行"
    
    # 最新のバックアップを検索
    LATEST_BACKUP=$(find backups/ -name "deploy_${DEPLOY_ENV}_*" -type d | sort -r | head -1)
    
    if [ -z "$LATEST_BACKUP" ]; then
        log_error "ロールバック用のバックアップが見つかりません"
        exit 1
    fi
    
    log_info "ロールバック対象: $LATEST_BACKUP"
    
    # アプリケーション停止
    stop_application
    
    # 設定ファイル復元
    if [ -d "$LATEST_BACKUP/config" ]; then
        cp -r "$LATEST_BACKUP/config/"* config/ 2>/dev/null || true
        log_success "設定ファイルを復元しました"
    fi
    
    # 環境変数ファイル復元
    if [ -f "$LATEST_BACKUP/.env.$DEPLOY_ENV" ]; then
        cp "$LATEST_BACKUP/.env.$DEPLOY_ENV" .
        log_success "環境変数ファイルを復元しました"
    fi
    
    # データベース復元（MongoDB）
    if [ -d "$LATEST_BACKUP/mongodb" ] && command -v mongorestore &> /dev/null; then
        log_info "データベースを復元中..."
        mongorestore --uri="$MONGODB_URI" --drop "$LATEST_BACKUP/mongodb" || log_warning "データベース復元に失敗"
    fi
    
    # アプリケーション再起動
    start_application
    
    if health_check; then
        log_success "✅ ロールバックが成功しました"
    else
        log_error "❌ ロールバック後のヘルスチェックが失敗しました"
        exit 1
    fi
}

# 使用方法表示
show_usage() {
    echo "使用方法: $0 [environment] [options]"
    echo ""
    echo "環境:"
    echo "  development    開発環境デプロイ"
    echo "  staging        ステージング環境デプロイ"
    echo "  production     本番環境デプロイ（デフォルト）"
    echo ""
    echo "オプション:"
    echo "  --rollback     最新バックアップからロールバック"
    echo "  --help         このヘルプを表示"
    echo ""
    echo "例:"
    echo "  $0 production                # 本番環境デプロイ"
    echo "  $0 staging                   # ステージング環境デプロイ"
    echo "  $0 production --rollback     # 本番環境ロールバック"
}

# メイン実行関数
main() {
    # 引数解析
    while [[ $# -gt 0 ]]; do
        case $1 in
            --rollback)
                ROLLBACK_MODE=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            development|staging|production)
                DEPLOY_ENV=$1
                shift
                ;;
            *)
                log_error "無効な引数: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    log_header "AI自動レポート生成システム デプロイ開始"
    log_info "環境: $DEPLOY_ENV"
    log_info "開始時刻: $(date)"
    
    # ロールバックモード
    if [ "$ROLLBACK_MODE" = true ]; then
        rollback
        exit 0
    fi
    
    # デプロイ確認（本番環境）
    if [ "$DEPLOY_ENV" = "production" ]; then
        echo ""
        log_warning "⚠️  本番環境へのデプロイを実行します"
        echo -n "続行しますか？ (y/N): "
        read -r CONFIRM
        if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
            log_info "デプロイをキャンセルしました"
            exit 0
        fi
    fi
    
    # デプロイ実行
    local attempt=1
    while [ $attempt -le $MAX_DEPLOY_ATTEMPTS ]; do
        log_info "デプロイ試行 $attempt/$MAX_DEPLOY_ATTEMPTS"
        
        if execute_deploy; then
            log_success "🎉 デプロイが正常に完了しました！"
            log_info "完了時刻: $(date)"
            
            # 成功通知
            if [ -n "$SLACK_WEBHOOK_URL" ]; then
                curl -X POST -H 'Content-type: application/json' \
                    --data "{\"text\":\"✅ AI Report System - $DEPLOY_ENV デプロイ成功 ($(date))\"}" \
                    "$SLACK_WEBHOOK_URL" 2>/dev/null || true
            fi
            
            exit 0
        else
            log_error "デプロイ試行 $attempt が失敗しました"
            if [ $attempt -eq $MAX_DEPLOY_ATTEMPTS ]; then
                log_error "❌ 全てのデプロイ試行が失敗しました"
                
                # 失敗通知
                if [ -n "$SLACK_WEBHOOK_URL" ]; then
                    curl -X POST -H 'Content-type: application/json' \
                        --data "{\"text\":\"❌ AI Report System - $DEPLOY_ENV デプロイ失敗 ($(date))\"}" \
                        "$SLACK_WEBHOOK_URL" 2>/dev/null || true
                fi
                
                exit 1
            fi
            
            log_info "$(( (MAX_DEPLOY_ATTEMPTS - attempt) * 30 ))秒後に再試行します..."
            sleep 30
        fi
        
        attempt=$((attempt + 1))
    done
}

# デプロイ実行（内部関数）
execute_deploy() {
    set -e
    
    load_deploy_config
    pre_deploy_checks
    create_backup
    stop_application
    update_dependencies
    build_application
    run_deployment_tests
    start_application
    
    if health_check; then
        post_deploy_tasks
        return 0
    else
        log_error "ヘルスチェック失敗。ロールバックを実行します..."
        rollback
        return 1
    fi
}

# スクリプト実行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
