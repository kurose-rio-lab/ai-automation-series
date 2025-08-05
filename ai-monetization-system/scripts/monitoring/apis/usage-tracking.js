/**
 * API使用量追跡ミドルウェア
 * RapidAPI Marketplace対応
 */
const express = require('express');
const redis = require('redis');
const mongoose = require('mongoose');

// Redis接続
const redisClient = redis.createClient({
  url: process.env.REDIS_URL
});

// 使用量スキーマ
const usageSchema = new mongoose.Schema({
  userId: { type: String, required: true },
  apiEndpoint: { type: String, required: true },
  timestamp: { type: Date, default: Date.now },
  responseTime: { type: Number },
  requestSize: { type: Number },
  responseSize: { type: Number },
  success: { type: Boolean, default: true },
  cost: { type: Number, default: 0 }
});

const Usage = mongoose.model('Usage', usageSchema);

/**
 * API使用量追跡ミドルウェア
 */
function trackUsage(req, res, next) {
  const startTime = Date.now();
  const userId = req.headers['x-rapidapi-user'] || req.headers['x-user-id'] || 'anonymous';
  const apiEndpoint = req.path;
  
  // レスポンス終了時の処理
  res.on('finish', async () => {
    const responseTime = Date.now() - startTime;
    const requestSize = JSON.stringify(req.body).length || 0;
    const responseSize = res.get('content-length') || 0;
    
    // 使用量データ作成
    const usageData = {
      userId,
      apiEndpoint,
      responseTime,
      requestSize,
      responseSize,
      success: res.statusCode < 400,
      cost: calculateApiCost(apiEndpoint, requestSize, responseSize)
    };
    
    try {
      // MongoDB に保存
      await new Usage(usageData).save();
      
      // Redis でリアルタイム集計
      await updateRealTimeStats(userId, apiEndpoint, usageData);
      
      // 使用量制限チェック
      await checkUsageLimits(userId);
      
    } catch (error) {
      console.error('使用量追跡エラー:', error);
    }
  });
  
  next();
}

/**
 * API コスト計算
 */
function calculateApiCost(endpoint, requestSize, responseSize) {
  const costRules = {
    '/api/analyze': 500,        // データ分析API: 500円
    '/api/generate-report': 2000, // レポート生成API: 2000円
    '/api/custom-analysis': 5000  // カスタム分析API: 5000円
  };
  
  const baseCost = costRules[endpoint] || 100;
  
  // データサイズに応じた追加コスト
  const sizeMultiplier = Math.max(1, (requestSize + responseSize) / 1024 / 1024); // MB単位
  
  return Math.round(baseCost * sizeMultiplier);
}

/**
 * リアルタイム統計更新
 */
async function updateRealTimeStats(userId, endpoint, usageData) {
  const today = new Date().toISOString().split('T')[0];
  const statsKey = `stats:${userId}:${today}`;
  
  await redisClient.hIncrBy(statsKey, 'total_requests', 1);
  await redisClient.hIncrBy(statsKey, 'total_cost', usageData.cost);
  await redisClient.hIncrBy(statsKey, `endpoint:${endpoint}`, 1);
  
  // TTL設定（7日間）
  await redisClient.expire(statsKey, 7 * 24 * 60 * 60);
}

/**
 * 使用量制限チェック
 */
async function checkUsageLimits(userId) {
  const today = new Date().toISOString().split('T')[0];
  const statsKey = `stats:${userId}:${today}`;
  
  const stats = await redisClient.hGetAll(statsKey);
  const totalCost = parseInt(stats.total_cost || 0);
  const totalRequests = parseInt(stats.total_requests || 0);
  
  // ユーザープランに応じた制限チェック
  const userPlan = await getUserPlan(userId);
  const limits = getPlanLimits(userPlan);
  
  // 制限超過チェック
  if (totalCost > limits.dailyCostLimit) {
    await sendLimitAlert(userId, 'cost', totalCost, limits.dailyCostLimit);
  }
  
  if (totalRequests > limits.dailyRequestLimit) {
    await sendLimitAlert(userId, 'requests', totalRequests, limits.dailyRequestLimit);
  }
}

/**
 * プラン制限設定
 */
function getPlanLimits(plan) {
  const limits = {
    'basic': {
      dailyRequestLimit: 100,
      dailyCostLimit: 50000  // 5万円
    },
    'pro': {
      dailyRequestLimit: 500,
      dailyCostLimit: 200000  // 20万円
    },
    'enterprise': {
      dailyRequestLimit: -1,  // 無制限
      dailyCostLimit: -1      // 無制限
    }
  };
  
  return limits[plan] || limits['basic'];
}

/**
 * 制限アラート送信
 */
async function sendLimitAlert(userId, type, current, limit) {
  const message = `⚠️ 使用量制限警告\nユーザー: ${userId}\n${type}: ${current}/${limit}`;
  
  // Slack通知
  await sendSlackAlert(message);
  
  // ユーザーへのメール通知
  await sendEmailAlert(userId, type, current, limit);
}

module.exports = { trackUsage, calculateApiCost };
