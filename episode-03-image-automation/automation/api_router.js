// API自動選択・ルーティングシステム
const axios = require('axios');

class APIRouter {
    constructor() {
        this.apiProviders = {
            midjourney: {
                name: 'Midjourney',
                costPerImage: 0.08,
                qualityScore: 9.5,
                speedScore: 6.0,
                maxConcurrent: 3,
                currentLoad: 0,
                capabilities: ['high_quality', 'artistic', 'detailed'],
                limitations: ['no_nsfw', 'english_only']
            },
            stable_diffusion: {
                name: 'Stable Diffusion',
                costPerImage: 0.01,
                qualityScore: 8.0,
                speedScore: 9.0,
                maxConcurrent: 10,
                currentLoad: 0,
                capabilities: ['fast', 'customizable', 'batch_processing'],
                limitations: []
            },
            dalle3: {
                name: 'DALL-E 3',
                costPerImage: 0.04,
                qualityScore: 9.0,
                speedScore: 7.0,
                maxConcurrent: 5,
                currentLoad: 0,
                capabilities: ['text_integration', 'detailed', 'coherent'],
                limitations: ['content_policy']
            }
        };
        
        this.requestHistory = [];
        this.performanceMetrics = {};
        
        // 初期化
        this.initializeMetrics();
    }
    
    initializeMetrics() {
        for (const provider in this.apiProviders) {
            this.performanceMetrics[provider] = {
                totalRequests: 0,
                successfulRequests: 0,
                averageResponseTime: 0,
                averageQualityScore: 0,
                uptime: 1.0,
                lastUpdated: new Date()
            };
        }
    }
    
    analyzeComplexity(prompt) {
        const complexKeywords = [
            'photorealistic', 'detailed', 'professional photography',
            'intricate', 'complex', 'elaborate', 'sophisticated'
        ];
        
        const simpleKeywords = [
            'simple', 'minimal', 'icon', 'logo', 'basic', 'clean'
        ];
        
        const promptLower = prompt.toLowerCase();
        
        const complexCount = complexKeywords.filter(keyword => 
            promptLower.includes(keyword)
        ).length;
        
        const simpleCount = simpleKeywords.filter(keyword => 
            promptLower.includes(keyword)
        ).length;
        
        if (complexCount > simpleCount) return 'complex';
        if (simpleCount > complexCount) return 'simple';
        return 'medium';
    }
    
    analyzeBudget(budgetLevel) {
        const budgetMapping = {
            'economy': { maxCost: 0.02, priority: 'cost' },
            'standard': { maxCost: 0.05, priority: 'balanced' },
            'premium': { maxCost: 0.20, priority: 'quality' }
        };
        
        return budgetMapping[budgetLevel] || budgetMapping['standard'];
    }
    
    analyzePriority(priority) {
        const priorityMapping = {
            'low': { urgency: 1, tolerance: 300 },      // 5分許容
            'normal': { urgency: 2, tolerance: 120 },   // 2分許容
            'high': { urgency: 3, tolerance: 60 },      // 1分許容
            'urgent': { urgency: 4, tolerance: 30 }     // 30秒許容
        };
        
        return priorityMapping[priority] || priorityMapping['normal'];
    }
    
    checkProviderAvailability(provider) {
        const providerData = this.apiProviders[provider];
        const metrics = this.performanceMetrics[provider];
        
        return {
            available: providerData.currentLoad < providerData.maxConcurrent,
            loadPercentage: (providerData.currentLoad / providerData.maxConcurrent) * 100,
            uptime: metrics.uptime,
            averageResponseTime: metrics.averageResponseTime
        };
    }
    
    calculateProviderScore(provider, request) {
        const providerData = this.apiProviders[provider];
        const metrics = this.performanceMetrics[provider];
        const availability = this.checkProviderAvailability(provider);
        
        if (!availability.available) {
            return 0; // 利用不可能
        }
        
        let score = 0;
        const weights = {
            cost: 0.25,
            quality: 0.30,
            speed: 0.20,
            availability: 0.15,
            reliability: 0.10
        };
        
        // コストスコア（安いほど高スコア）
        const maxCost = 0.20;
        const costScore = (maxCost - providerData.costPerImage) / maxCost;
        score += costScore * weights.cost;
        
        // 品質スコア
        const qualityScore = providerData.qualityScore / 10;
        score += qualityScore * weights.quality;
        
        // 速度スコア
        const speedScore = providerData.speedScore / 10;
        score += speedScore * weights.speed;
        
        // 可用性スコア
        const availabilityScore = 1 - (availability.loadPercentage / 100);
        score += availabilityScore * weights.availability;
        
        // 信頼性スコア
        const reliabilityScore = metrics.uptime * (metrics.successfulRequests / Math.max(metrics.totalRequests, 1));
        score += reliabilityScore * weights.reliability;
        
        // 要求特性に基づく調整
        if (request.complexity === 'complex' && providerData.capabilities.includes('detailed')) {
            score *= 1.2;
        }
        
        if (request.budget === 'economy' && providerData.costPerImage < 0.02) {
            score *= 1.3;
        }
        
        if (request.priority === 'urgent' && providerData.speedScore > 8) {
            score *= 1.4;
        }
        
        return Math.min(score, 1.0);
    }
    
    selectOptimalProvider(request) {
        const {
            prompt,
            budget = 'standard',
            priority = 'normal',
            quality_preference = 'balanced',
            complexity = null
        } = request;
        
        // 分析
        const analyzedComplexity = complexity || this.analyzeComplexity(prompt);
        const budgetInfo = this.analyzeBudget(budget);
        const priorityInfo = this.analyzePriority(priority);
        
        const analysisResult = {
            complexity: analyzedComplexity,
            budget: budgetInfo,
            priority: priorityInfo
        };
        
        // 各プロバイダーのスコア計算
        const providerScores = {};
        
        for (const provider in this.apiProviders) {
            const requestData = {
                ...request,
                complexity: analyzedComplexity,
                budget: budget,
                priority: priority
            };
            
            providerScores[provider] = this.calculateProviderScore(provider, requestData);
        }
        
        // 最適プロバイダー選択
        const bestProvider = Object.entries(providerScores).reduce((best, [provider, score]) => {
            if (score > best.score) {
                return { provider, score };
            }
            return best;
        }, { provider: null, score: 0 });
        
        // フォールバック処理
        if (bestProvider.score === 0) {
            // 全てのプロバイダーが利用不可能な場合
            const fallbackProvider = this.findFallbackProvider();
            return {
                provider: fallbackProvider,
                score: 0,
                reason: 'fallback',
                analysis: analysisResult,
                allScores: providerScores
            };
        }
        
        return {
            provider: bestProvider.provider,
            score: bestProvider.score,
            reason: 'optimal',
            analysis: analysisResult,
            allScores: providerScores
        };
    }
    
    findFallbackProvider() {
        // 負荷が最も低いプロバイダーを選択
        let fallbackProvider = null;
        let minLoad = Infinity;
        
        for (const provider in this.apiProviders) {
            const providerData = this.apiProviders[provider];
            if (providerData.currentLoad < minLoad) {
                minLoad = providerData.currentLoad;
                fallbackProvider = provider;
            }
        }
        
        return fallbackProvider || 'stable_diffusion'; // 最終フォールバック
    }
    
    async routeRequest(request) {
        const startTime = Date.now();
        
        try {
            // プロバイダー選択
            const selection = this.selectOptimalProvider(request);
            
            if (!selection.provider) {
                throw new Error('利用可能なプロバイダーがありません');
            }
            
            console.log(`API選択: ${selection.provider} (スコア: ${selection.score.toFixed(3)})`);
            
            // 負荷カウント増加
            this.apiProviders[selection.provider].currentLoad++;
            
            // リクエスト実行
            const result = await this.executeRequest(selection.provider, request);
            
            // 成功時の処理
            const responseTime = Date.now() - startTime;
            this.updateMetrics(selection.provider, true, responseTime, result.qualityScore);
            
            // 履歴保存
            this.requestHistory.push({
                timestamp: new Date(),
                provider: selection.provider,
                request: request,
                success: true,
                responseTime: responseTime,
                selection: selection
            });
            
            return {
                success: true,
                provider: selection.provider,
                result: result,
                selection: selection,
                responseTime: responseTime
            };
            
        } catch (error) {
            // エラー時の処理
            const responseTime = Date.now() - startTime;
            
            if (selection && selection.provider) {
                this.updateMetrics(selection.provider, false, responseTime, 0);
            }
            
            // 履歴保存
            this.requestHistory.push({
                timestamp: new Date(),
                provider: selection ? selection.provider : 'unknown',
                request: request,
                success: false,
                error: error.message,
                responseTime: responseTime,
                selection: selection
            });
            
            return {
                success: false,
                error: error.message,
                selection: selection,
                responseTime: responseTime
            };
            
        } finally {
            // 負荷カウント減少
            if (selection && selection.provider) {
                this.apiProviders[selection.provider].currentLoad--;
            }
        }
    }
    
    async executeRequest(provider, request) {
        // プロバイダー別のAPI呼び出し
        switch (provider) {
            case 'midjourney':
                return await this.callMidjourneyAPI(request);
            case 'stable_diffusion':
                return await this.callStableDiffusionAPI(request);
            case 'dalle3':
                return await this.callDALLE3API(request);
            default:
                throw new Error(`未対応のプロバイダー: ${provider}`);
        }
    }
    
    async callMidjourneyAPI(request) {
        const response = await axios.post('https://api.midjourney.com/v1/imagine', {
            prompt: request.prompt,
            aspect_ratio: `${request.width || 512}:${request.height || 512}`,
            quality: request.quality || 2
        }, {
            headers: {
                'Authorization': `Bearer ${process.env.MIDJOURNEY_API_KEY}`,
                'Content-Type': 'application/json'
            }
        });
        
        return {
            imageUrl: response.data.image_url,
            qualityScore: 9.5,
            metadata: response.data
        };
    }
    
    async callStableDiffusionAPI(request) {
        // Stable Diffusion API呼び出し（実装例）
        const response = await axios.post('http://localhost:7860/sdapi/v1/txt2img', {
            prompt: request.prompt,
            negative_prompt: request.negative_prompt || '',
            width: request.width || 512,
            height: request.height || 512,
            steps: request.steps || 20,
            cfg_scale: request.guidance_scale || 7.5
        });
        
        return {
            imageBase64: response.data.images[0],
            qualityScore: 8.0,
            metadata: response.data.info
        };
    }
    
    async callDALLE3API(request) {
        const response = await axios.post('https://api.openai.com/v1/images/generations', {
            prompt: request.prompt,
            n: 1,
            size: `${request.width || 512}x${request.height || 512}`,
            quality: request.quality || 'standard'
        }, {
            headers: {
                'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
                'Content-Type': 'application/json'
            }
        });
        
        return {
            imageUrl: response.data.data[0].url,
            qualityScore: 9.0,
            metadata: response.data
        };
    }
    
    updateMetrics(provider, success, responseTime, qualityScore) {
        const metrics = this.performanceMetrics[provider];
        
        metrics.totalRequests++;
        
        if (success) {
            metrics.successfulRequests++;
        }
        
        // 平均応答時間更新
        const totalTime = metrics.averageResponseTime * (metrics.totalRequests - 1);
        metrics.averageResponseTime = (totalTime + responseTime) / metrics.totalRequests;
        
        // 平均品質スコア更新
        if (success && qualityScore > 0) {
            const totalQuality = metrics.averageQualityScore * (metrics.successfulRequests - 1);
            metrics.averageQualityScore = (totalQuality + qualityScore) / metrics.successfulRequests;
        }
        
        // アップタイム更新
        metrics.uptime = metrics.successfulRequests / metrics.totalRequests;
        
        metrics.lastUpdated = new Date();
    }
    
    getProviderStats() {
        return {
            providers: this.apiProviders,
            metrics: this.performanceMetrics,
            totalRequests: this.requestHistory.length,
            successRate: this.requestHistory.filter(r => r.success).length / this.requestHistory.length
        };
    }
    
    getRecommendations() {
        const recommendations = [];
        
        for (const provider in this.performanceMetrics) {
            const metrics = this.performanceMetrics[provider];
            
            if (metrics.uptime < 0.9) {
                recommendations.push({
                    type: 'reliability',
                    provider: provider,
                    message: `${provider}の信頼性が低下しています (${(metrics.uptime * 100).toFixed(1)}%)`
                });
            }
            
            if (metrics.averageResponseTime > 120000) { // 2分以上
                recommendations.push({
                    type: 'performance',
                    provider: provider,
                    message: `${provider}の応答時間が遅くなっています (${(metrics.averageResponseTime / 1000).toFixed(1)}s)`
                });
            }
        }
        
        return recommendations;
    }
}

// 使用例
const router = new APIRouter();

// テストリクエスト
const testRequest = {
    prompt: "A beautiful sunset over mountains",
    budget: "standard",
    priority: "normal",
    width: 512,
    height: 512
};

router.routeRequest(testRequest)
    .then(result => {
        console.log('ルーティング結果:', result);
    })
    .catch(error => {
        console.error('エラー:', error);
    });

module.exports = APIRouter;
