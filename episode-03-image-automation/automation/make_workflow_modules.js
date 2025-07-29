// Make.com ワークフロー制御モジュール
class MakeWorkflowModules {
    constructor() {
        this.styleMap = {
            'professional': 'clean, corporate, professional photography',
            'creative': 'artistic, creative, innovative design',
            'minimal': 'minimal, clean, modern aesthetic',
            'vibrant': 'colorful, energetic, dynamic',
            'elegant': 'sophisticated, refined, luxury'
        };
        
        this.brandElements = {
            'tech_company': {
                colorScheme: 'blue and white corporate colors',
                typography: 'modern sans-serif',
                mood: 'professional and innovative'
            },
            'fashion': {
                colorScheme: 'elegant monochrome with gold accents',
                typography: 'sophisticated serif',
                mood: 'luxurious and stylish'
            },
            'food_service': {
                colorScheme: 'warm and appetizing colors',
                typography: 'friendly and approachable',
                mood: 'delicious and inviting'
            },
            'healthcare': {
                colorScheme: 'clean whites and medical blues',
                typography: 'trustworthy and clear',
                mood: 'professional and caring'
            },
            'education': {
                colorScheme: 'friendly and approachable colors',
                typography: 'clear and readable',
                mood: 'educational and inspiring'
            }
        };
        
        this.aspectRatios = {
            'social_media': '1:1',
            'instagram_story': '9:16',
            'facebook_cover': '851:315',
            'youtube_thumbnail': '16:9',
            'website_banner': '16:9',
            'mobile_app': '9:16',
            'print_flyer': '8.5:11',
            'business_card': '3.5:2'
        };
    }
    
    // データ検証モジュール
    validateData(input) {
        const errors = [];
        
        // 必須フィールドチェック
        if (!input.prompt || typeof input.prompt !== 'string') {
            errors.push('プロンプトは必須です');
        }
        
        if (!input.style || typeof input.style !== 'string') {
            errors.push('スタイルは必須です');
        }
        
        if (!input.aspect_ratio || typeof input.aspect_ratio !== 'string') {
            errors.push('アスペクト比は必須です');
        }
        
        // プロンプト長制限
        if (input.prompt && input.prompt.length > 1000) {
            input.prompt = input.prompt.substring(0, 1000) + '...';
        }
        
        // スタイル有効性チェック
        if (input.style && !this.styleMap[input.style]) {
            errors.push(`無効なスタイル: ${input.style}`);
        }
        
        // アスペクト比有効性チェック
        if (input.aspect_ratio && !this.isValidAspectRatio(input.aspect_ratio)) {
            errors.push(`無効なアスペクト比: ${input.aspect_ratio}`);
        }
        
        if (errors.length > 0) {
            throw new Error(`データ検証エラー: ${errors.join(', ')}`);
        }
        
        return input;
    }
    
    // アスペクト比検証
    isValidAspectRatio(ratio) {
        const validRatios = Object.values(this.aspectRatios);
        return validRatios.includes(ratio) || /^\d+:\d+$/.test(ratio);
    }
    
    // ブランド要素取得
    getBrandElements(brand) {
        return this.brandElements[brand] || {
            colorScheme: 'balanced color palette',
            typography: 'clean typography',
            mood: 'professional'
        };
    }
    
    // カテゴリ別アスペクト比取得
    getAspectRatio(category) {
        return this.aspectRatios[category] || '1:1';
    }
    
    // 動的プロンプト生成
    buildPrompt(basePrompt, style, brand, category) {
        let enhancedPrompt = basePrompt;
        
        // スタイル追加
        if (style && this.styleMap[style]) {
            enhancedPrompt += `, ${this.styleMap[style]}`;
        }
        
        // ブランド要素追加
        if (brand) {
            const brandElements = this.getBrandElements(brand);
            enhancedPrompt += `, ${brandElements.colorScheme}, ${brandElements.typography}, ${brandElements.mood}`;
        }
        
        // カテゴリ特有の要素追加
        if (category) {
            const categoryEnhancements = this.getCategoryEnhancements(category);
            enhancedPrompt += `, ${categoryEnhancements}`;
        }
        
        // 品質向上キーワード追加
        enhancedPrompt += ', high quality, detailed, professional';
        
        // アスペクト比設定
        const aspectRatio = this.getAspectRatio(category);
        enhancedPrompt += ` --ar ${aspectRatio} --q 2`;
        
        return enhancedPrompt;
    }
    
    // カテゴリ特有の強化要素
    getCategoryEnhancements(category) {
        const enhancements = {
            'social_media': 'engaging, eye-catching, social media optimized',
            'instagram_story': 'vertical format, story-friendly, mobile optimized',
            'facebook_cover': 'cover photo style, wide format, brand representative',
            'youtube_thumbnail': 'thumbnail style, attention-grabbing, clickable',
            'website_banner': 'web banner style, professional, brand-aligned',
            'mobile_app': 'app-friendly, mobile UI, clean interface',
            'print_flyer': 'print-ready, high resolution, marketing material',
            'business_card': 'business card format, professional, contact information ready'
        };
        
        return enhancements[category] || 'professional, high quality';
    }
    
    // 複雑度分析
    analyzeComplexity(prompt) {
        const complexityIndicators = {
            high: ['photorealistic', 'detailed', 'intricate', 'complex scene', 'multiple objects'],
            medium: ['professional', 'modern', 'stylized', 'artistic'],
            low: ['simple', 'minimal', 'icon', 'logo', 'basic']
        };
        
        const promptLower = prompt.toLowerCase();
        let complexity = 'medium'; // デフォルト
        
        for (const [level, keywords] of Object.entries(complexityIndicators)) {
            const matchCount = keywords.filter(keyword => promptLower.includes(keyword)).length;
            if (matchCount > 0) {
                complexity = level;
                break;
            }
        }
        
        return complexity;
    }
    
    // API選択ロジック
    selectAPI(prompt, budget, priority) {
        const complexity = this.analyzeComplexity(prompt);
        
        // 予算とプライオリティに基づく選択
        if (priority === 'high' && budget === 'premium') {
            return 'midjourney';
        } else if (complexity === 'low' || budget === 'economy') {
            return 'stable_diffusion';
        } else if (complexity === 'high' && budget !== 'economy') {
            return 'midjourney';
        } else {
            return 'stable_diffusion';
        }
    }
    
    // 品質設定決定
    determineQualitySettings(api, priority, budget) {
        const settings = {
            midjourney: {
                economy: { quality: 1, stylize: 400 },
                standard: { quality: 1, stylize: 750 },
                premium: { quality: 2, stylize: 1000 }
            },
            stable_diffusion: {
                economy: { steps: 15, guidance_scale: 7.0 },
                standard: { steps: 20, guidance_scale: 7.5 },
                premium: { steps: 30, guidance_scale: 8.0 }
            },
            dalle3: {
                economy: { quality: 'standard' },
                standard: { quality: 'standard' },
                premium: { quality: 'hd' }
            }
        };
        
        return settings[api][budget] || settings[api]['standard'];
    }
    
    // エラーハンドリング
    handleError(error, context) {
        const errorTypes = {
            'validation': {
                severity: 'medium',
                retry: false,
                message: 'データ検証エラーが発生しました'
            },
            'api': {
                severity: 'high',
                retry: true,
                message: 'API呼び出しでエラーが発生しました'
            },
            'timeout': {
                severity: 'medium',
                retry: true,
                message: 'タイムアウトが発生しました'
            },
            'quota': {
                severity: 'high',
                retry: false,
                message: 'APIクォータを超過しました'
            }
        };
        
        const errorType = this.categorizeError(error);
        const errorInfo = errorTypes[errorType] || errorTypes['api'];
        
        return {
            error: error.message,
            type: errorType,
            severity: errorInfo.severity,
            shouldRetry: errorInfo.retry,
            context: context,
            timestamp: new Date().toISOString(),
            recommendations: this.getErrorRecommendations(errorType)
        };
    }
    
    // エラー分類
    categorizeError(error) {
        const errorMessage = error.message.toLowerCase();
        
        if (errorMessage.includes('validation') || errorMessage.includes('invalid')) {
            return 'validation';
        } else if (errorMessage.includes('timeout')) {
            return 'timeout';
        } else if (errorMessage.includes('quota') || errorMessage.includes('limit')) {
            return 'quota';
        } else {
            return 'api';
        }
    }
    
    // エラー対応推奨事項
    getErrorRecommendations(errorType) {
        const recommendations = {
            'validation': [
                '入力データの形式を確認してください',
                '必須フィールドが全て入力されているか確認してください'
            ],
            'api': [
                'APIキーが正しく設定されているか確認してください',
                'ネットワーク接続を確認してください',
                '少し時間をおいてから再試行してください'
            ],
            'timeout': [
                'タイムアウト時間を延長してください',
                'リクエストを小さな単位に分割してください'
            ],
            'quota': [
                'APIクォータの使用量を確認してください',
                '別のAPIプロバイダーを検討してください',
                'プランのアップグレードを検討してください'
            ]
        };
        
        return recommendations[errorType] || ['サポートにお問い合わせください'];
    }
    
    // 処理結果のフィルタリング
    filterResults(results, criteria) {
        return results.filter(result => {
            if (criteria.minQuality && result.qualityScore < criteria.minQuality) {
                return false;
            }
            
            if (criteria.maxProcessingTime && result.processingTime > criteria.maxProcessingTime) {
                return false;
            }
            
            if (criteria.requiredBrand && result.brand !== criteria.requiredBrand) {
                return false;
            }
            
            if (criteria.excludeErrors && !result.success) {
                return false;
            }
            
            return true;
        });
    }
    
    // 処理統計生成
    generateProcessingStats(results) {
        const stats = {
            totalProcessed: results.length,
            successful: results.filter(r => r.success).length,
            failed: results.filter(r => !r.success).length,
            averageProcessingTime: 0,
            averageQualityScore: 0,
            apiUsage: {},
            brandDistribution: {},
            categoryDistribution: {}
        };
        
        const successfulResults = results.filter(r => r.success);
        
        if (successfulResults.length > 0) {
            stats.averageProcessingTime = successfulResults.reduce((sum, r) => sum + r.processingTime, 0) / successfulResults.length;
            stats.averageQualityScore = successfulResults.reduce((sum, r) => sum + (r.qualityScore || 0), 0) / successfulResults.length;
        }
        
        // API使用統計
        results.forEach(result => {
            const api = result.apiUsed || 'unknown';
            stats.apiUsage[api] = (stats.apiUsage[api] || 0) + 1;
        });
        
        // ブランド分布
        results.forEach(result => {
            const brand = result.brand || 'unknown';
            stats.brandDistribution[brand] = (stats.brandDistribution[brand] || 0) + 1;
        });
        
        // カテゴリ分布
        results.forEach(result => {
            const category = result.category || 'unknown';
            stats.categoryDistribution[category] = (stats.categoryDistribution[category] || 0) + 1;
        });
        
        return stats;
    }
    
    // Make.com用のレスポンス形式
    formatMakeResponse(success, data, error = null) {
        const response = {
            success: success,
            timestamp: new Date().toISOString(),
            data: data,
            error: error
        };
        
        // Make.comが期待する形式に調整
        if (success) {
            response.status = 'completed';
            response.output = data;
        } else {
            response.status = 'failed';
            response.error_message = error;
        }
        
        return response;
    }
}

// 使用例とテスト
const workflowModules = new MakeWorkflowModules();

// テストデータ
const testInput = {
    prompt: "A modern office space",
    style: "professional",
    brand: "tech_company",
    category: "social_media",
    aspect_ratio: "1:1",
    budget: "standard",
    priority: "normal"
};

try {
    // データ検証
    const validatedInput = workflowModules.validateData(testInput);
    console.log("✓ データ検証完了");
    
    // プロンプト生成
    const enhancedPrompt = workflowModules.buildPrompt(
        validatedInput.prompt,
        validatedInput.style,
        validatedInput.brand,
        validatedInput.category
    );
    console.log("✓ プロンプト生成完了:", enhancedPrompt);
    
    // API選択
    const selectedAPI = workflowModules.selectAPI(
        validatedInput.prompt,
        validatedInput.budget,
        validatedInput.priority
    );
    console.log("✓ API選択完了:", selectedAPI);
    
    // 品質設定
    const qualitySettings = workflowModules.determineQualitySettings(
        selectedAPI,
        validatedInput.priority,
        validatedInput.budget
    );
    console.log("✓ 品質設定完了:", qualitySettings);
    
} catch (error) {
    const errorInfo = workflowModules.handleError(error, testInput);
    console.error("✗ エラー発生:", errorInfo);
}

module.exports = MakeWorkflowModules;
