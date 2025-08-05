/**
 * 設定データの自動バックアップシステム
 * 毎日午前2時に実行されるトリガー設定
 */
function backupConfiguration() {
  try {
    // バックアップフォルダの取得
    const backupFolder = DriveApp.getFolderById('your_backup_folder_id');
    
    // 現在の設定スプレッドシートを取得
    const configSheet = SpreadsheetApp.getActiveSpreadsheet();
    
    // タイムスタンプ付きでコピー作成
    const timestamp = new Date().toISOString().split('T')[0];
    const backupName = `Config_Backup_${timestamp}`;
    const backupFile = configSheet.copy(backupName);
    
    // バックアップフォルダに移動
    DriveApp.getFileById(backupFile.getId()).moveTo(backupFolder);
    
    // 古いバックアップファイルを削除（30日以前）
    cleanOldBackups(backupFolder);
    
    // 成功ログ
    console.log(`バックアップ完了: ${backupName}`);
    
    // Slack通知（オプション）
    sendSlackNotification(`✅ 設定バックアップが完了しました: ${backupName}`);
    
  } catch (error) {
    console.error('バックアップエラー:', error);
    sendSlackNotification(`❌ バックアップエラー: ${error.message}`);
  }
}

/**
 * 古いバックアップファイルの削除
 */
function cleanOldBackups(folder) {
  const files = folder.getFiles();
  const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
  
  while (files.hasNext()) {
    const file = files.next();
    if (file.getDateCreated() < thirtyDaysAgo && file.getName().includes('Config_Backup_')) {
      file.setTrashed(true);
      console.log(`古いバックアップを削除: ${file.getName()}`);
    }
  }
}

/**
 * Slack通知送信
 */
function sendSlackNotification(message) {
  const webhookUrl = 'your_slack_webhook_url';
  const payload = {
    'text': message,
    'username': 'AI Report System',
    'icon_emoji': ':robot_face:'
  };
  
  const options = {
    'method': 'POST',
    'contentType': 'application/json',
    'payload': JSON.stringify(payload)
  };
  
  try {
    UrlFetchApp.fetch(webhookUrl, options);
  } catch (error) {
    console.error('Slack通知エラー:', error);
  }
}

/**
 * データベースバックアップ（MongoDB）
 */
function backupDatabase() {
  const scriptProperties = PropertiesService.getScriptProperties();
  const mongoUri = scriptProperties.getProperty('MONGODB_URI');
  
  if (!mongoUri) {
    console.error('MongoDB URI が設定されていません');
    return;
  }
  
  // MongoDBデータのエクスポート処理
  // 実際の実装では、MongoDB APIまたは外部サービスを使用
  
  console.log('データベースバックアップ完了');
}
