/**
 * Google Apps Script - データクリーニング自動化システム
 * レポート生成用データの品質保証とクリーニング
 */

function main() {
  Logger.log('データクリーニング処理開始');
  
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  
  // 各シートのクリーニング実行
  cleanSalesData(spreadsheet);
  cleanCustomerData(spreadsheet);
  cleanMarketingData(spreadsheet);
  
  // データ統合と品質チェック
  validateDataIntegrity(spreadsheet);
  
  Logger.log('データクリーニング処理完了');
}

function cleanSalesData(spreadsheet) {
  const sheet = spreadsheet.getSheetByName('売上データ');
  if (!sheet) return;
  
  const range = sheet.getDataRange();
  const values = range.getValues();
  const headers = values[0];
  
  // ヘッダー行をスキップして処理
  for (let i = 1; i < values.length; i++) {
    const row = values[i];
    
    // 売上金額の数値化と検証
    if (typeof row[getColumnIndex(headers, '売上金額')] === 'string') {
      row[getColumnIndex(headers, '売上金額')] = parseFloat(
        row[getColumnIndex(headers, '売上金額')].toString().replace(/[^\d.-]/g, '')
      ) || 0;
    }
    
    // 日付の正規化
    const dateIndex = getColumnIndex(headers, '日付');
    if (dateIndex !== -1 && row[dateIndex]) {
      try {
        row[dateIndex] = new Date(row[dateIndex]);
      } catch (e) {
        row[dateIndex] = new Date();
        Logger.log(`日付変換エラー: 行${i + 1}`);
      }
    }
    
    // 商品名の正規化
    const productIndex = getColumnIndex(headers, '商品名');
    if (productIndex !== -1 && row[productIndex]) {
      row[productIndex] = row[productIndex].toString().trim();
    }
    
    // 重複チェックとマーキング
    if (isDuplicateRow(values, i, ['日付', '商品名', '売上金額'])) {
      sheet.getRange(i + 1, 1, 1, row.length).setBackground('#FFE6E6');
    }
  }
  
  // クリーニング結果を反映
  range.setValues(values);
}

function cleanCustomerData(spreadsheet) {
  const sheet = spreadsheet.getSheetByName('顧客データ');
  if (!sheet) return;
  
  const range = sheet.getDataRange();
  const values = range.getValues();
  const headers = values[0];
  
  for (let i = 1; i < values.length; i++) {
    const row = values[i];
    
    // メールアドレスの検証
    const emailIndex = getColumnIndex(headers, 'メールアドレス');
    if (emailIndex !== -1 && row[emailIndex]) {
      const email = row[emailIndex].toString().trim().toLowerCase();
      if (!isValidEmail(email)) {
        sheet.getRange(i + 1, emailIndex + 1).setBackground('#FFE6E6');
        sheet.getRange(i + 1, emailIndex + 1).setNote('無効なメールアドレス');
      }
      row[emailIndex] = email;
    }
    
    // 電話番号の正規化
    const phoneIndex = getColumnIndex(headers, '電話番号');
    if (phoneIndex !== -1 && row[phoneIndex]) {
      row[phoneIndex] = normalizePhoneNumber(row[phoneIndex].toString());
    }
    
    // 顧客ステータスの統一
    const statusIndex = getColumnIndex(headers, 'ステータス');
    if (statusIndex !== -1 && row[statusIndex]) {
      row[statusIndex] = normalizeCustomerStatus(row[statusIndex].toString());
    }
  }
  
  range.setValues(values);
}

function cleanMarketingData(spreadsheet) {
  const sheet = spreadsheet.getSheetByName('マーケティングデータ');
  if (!sheet) return;
  
  const range = sheet.getDataRange();
  const values = range.getValues();
  const headers = values[0];
  
  for (let i = 1; i < values.length; i++) {
    const row = values[i];
    
    // コンバージョン率の正規化
    const cvRateIndex = getColumnIndex(headers, 'コンバージョン率');
    if (cvRateIndex !== -1 && row[cvRateIndex]) {
      let cvRate = parseFloat(row[cvRateIndex].toString().replace('%', ''));
      if (cvRate > 1) cvRate = cvRate / 100; // パーセント表記を小数に変換
      row[cvRateIndex] = cvRate;
    }
    
    // CPCの数値化
    const cpcIndex = getColumnIndex(headers, 'CPC');
    if (cpcIndex !== -1 && row[cpcIndex]) {
      row[cpcIndex] = parseFloat(
        row[cpcIndex].toString().replace(/[^\d.-]/g, '')
      ) || 0;
    }
    
    // キャンペーン名の正規化
    const campaignIndex = getColumnIndex(headers, 'キャンペーン名');
    if (campaignIndex !== -1 && row[campaignIndex]) {
      row[campaignIndex] = row[campaignIndex].toString().trim();
    }
  }
  
  range.setValues(values);
}

function validateDataIntegrity(spreadsheet) {
  const report = [];
  
  // 各シートのデータ整合性チェック
  const sheets = ['売上データ', '顧客データ', 'マーケティングデータ'];
  
  sheets.forEach(sheetName => {
    const sheet = spreadsheet.getSheetByName(sheetName);
    if (!sheet) {
      report.push(`エラー: ${sheetName}シートが見つかりません`);
      return;
    }
    
    const dataRange = sheet.getDataRange();
    const rowCount = dataRange.getNumRows();
    const colCount = dataRange.getNumColumns();
    
    if (rowCount <= 1) {
      report.push(`警告: ${sheetName}にデータがありません`);
    }
    
    // 空白行の検出
    const values = dataRange.getValues();
    let emptyRows = 0;
    for (let i = 1; i < values.length; i++) {
      if (values[i].every(cell => !cell || cell.toString().trim() === '')) {
        emptyRows++;
      }
    }
    
    if (emptyRows > 0) {
      report.push(`警告: ${sheetName}に${emptyRows}個の空白行があります`);
    }
    
    report.push(`完了: ${sheetName} - ${rowCount - 1}行のデータを処理`);
  });
  
  // レポート出力
  const reportSheet = getOrCreateSheet(spreadsheet, 'データ品質レポート');
  const reportData = [['タイムスタンプ', 'ステータス']];
  report.forEach(item => {
    reportData.push([new Date(), item]);
  });
  
  reportSheet.clear();
  reportSheet.getRange(1, 1, reportData.length, 2).setValues(reportData);
}

// ユーティリティ関数
function getColumnIndex(headers, columnName) {
  return headers.indexOf(columnName);
}

function isDuplicateRow(values, currentIndex, keyColumns) {
  const currentRow = values[currentIndex];
  const headers = values[0];
  
  for (let i = 1; i < values.length; i++) {
    if (i === currentIndex) continue;
    
    const compareRow = values[i];
    let isDuplicate = true;
    
    keyColumns.forEach(colName => {
      const colIndex = getColumnIndex(headers, colName);
      if (colIndex !== -1 && currentRow[colIndex] !== compareRow[colIndex]) {
        isDuplicate = false;
      }
    });
    
    if (isDuplicate) return true;
  }
  
  return false;
}

function isValidEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

function normalizePhoneNumber(phone) {
  // 日本の電話番号形式に正規化
  const cleaned = phone.replace(/[^\d-]/g, '');
  if (cleaned.match(/^\d{2,4}-\d{2,4}-\d{4}$/)) {
    return cleaned;
  }
  return phone; // 正規化できない場合は元の値を返す
}

function normalizeCustomerStatus(status) {
  const statusMap = {
    '新規': 'NEW',
    'アクティブ': 'ACTIVE',
    '休眠': 'DORMANT',
    '解約': 'CANCELLED'
  };
  
  return statusMap[status] || status;
}

function getOrCreateSheet(spreadsheet, sheetName) {
  let sheet = spreadsheet.getSheetByName(sheetName);
  if (!sheet) {
    sheet = spreadsheet.insertSheet(sheetName);
  }
  return sheet;
}

// 定期実行用のトリガー設定
function createTriggers() {
  // 既存のトリガーを削除
  const triggers = ScriptApp.getProjectTriggers();
  triggers.forEach(trigger => ScriptApp.deleteTrigger(trigger));
  
  // 毎日午前2時に実行
  ScriptApp.newTrigger('main')
    .timeBased()
    .everyDays(1)
    .atHour(2)
    .create();
    
  Logger.log('定期実行トリガーを設定しました');
}
