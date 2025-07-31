// runway_automation_workflow.js
// Make.com高度ワークフロー設計システム

class RunwayWorkflowManager {
    constructor(config) {
        this.apiKey = config.apiKey;
        this.baseURL = 'https://api.runwayml.com/v1';
        this.maxRetries = 3;
        this.requestQueue = [];
        this.processing = false;
    }

    async processVideoRequest(request) {
        try {
            console.log(`Processing video request: ${request.id}`);
            
            // Step 1: 動画タイプ別テンプレート選択
            const template = this.selectTemplate(request.type);
            
            // Step 2: 前半画像システムから素材取得
            const imageAssets = await this.getImageAssets(request.projectId);
            
            // Step 3: RunwayML API呼び出し・動画生成
            const videoJob = await this.generateVideo(template, imageAssets, request);
            
            // Step 4: 生成進捗監視
            const completedVideo = await this.monitorVideoGeneration(videoJob.id);
            
            return {
                success: true,
                videoPath: completedVideo.outputPath,
                generationTime: completedVideo.processingTime,
                costs: completedVideo.apiCosts
            };
            
        } catch (error) {
            console.error(`Video generation failed: ${error.message}`);
            return {
                success: false,
                error: error.message,
                retryable: this.isRetryableError(error)
            };
        }
    }

    selectTemplate(videoType) {
        const templates = {
            'product_showcase': {
                duration: 30,
                style: 'professional_clean',
                transitions: ['fade', 'slide'],
                cameraMovements: ['pan', 'zoom', 'static']
            },
            'social_story': {
                duration: 15,
                style: 'dynamic_engaging',
                transitions: ['quick_cut', 'zoom'],
                cameraMovements: ['handheld', 'tracking']
            },
            'explainer_video': {
                duration: 60,
                style: 'educational_clear',
                transitions: ['dissolve', 'wipe'],
                cameraMovements: ['steady', 'slow_pan']
            }
        };
        
        return templates[videoType] || templates['product_showcase'];
    }

    async getImageAssets(projectId) {
        // 前半画像システムとの統合
        const response = await fetch(`/api/projects/${projectId}/images`);
        const imageData = await response.json();
        
        return imageData.images.map(img => ({
            path: img.filePath,
            type: img.category,
            qualityScore: img.qualityScore,
            metadata: img.metadata
        }));
    }

    async generateVideo(template, imageAssets, request) {
        const prompt = this.buildPrompt(template, request);
        
        const apiRequest = {
            prompt: prompt,
            duration: template.duration,
            aspect_ratio: request.aspectRatio || '16:9',
            style: template.style,
            seed: request.seed || Math.floor(Math.random() * 1000000),
            image_assets: imageAssets.slice(0, 4) // RunwayML制限
        };

        const response = await fetch(`${this.baseURL}/generate`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(apiRequest)
        });

        if (!response.ok) {
            throw new Error(`RunwayML API error: ${response.status}`);
        }

        return await response.json();
    }

    buildPrompt(template, request) {
        const basePrompt = request.prompt || '';
        const styleModifiers = {
            'professional_clean': 'clean corporate aesthetic, professional lighting, minimal design',
            'dynamic_engaging': 'vibrant energetic style, dynamic camera work, engaging visuals',
            'educational_clear': 'clear instructional style, steady presentation, informative graphics'
        };

        return `${basePrompt}, ${styleModifiers[template.style]}, ${template.duration} seconds, high quality, commercial grade`;
    }

    async monitorVideoGeneration(jobId) {
        let attempts = 0;
        const maxAttempts = 60; // 最大10分待機
        
        while (attempts < maxAttempts) {
            try {
                const response = await fetch(`${this.baseURL}/jobs/${jobId}`, {
                    headers: {
                        'Authorization': `Bearer ${this.apiKey}`
                    }
                });
                
                const job = await response.json();
                
                if (job.status === 'completed') {
                    return {
                        outputPath: job.output_url,
                        processingTime: job.processing_time,
                        apiCosts: job.cost_breakdown
                    };
                } else if (job.status === 'failed') {
                    throw new Error(`Video generation failed: ${job.error}`);
                }
                
                // 10秒待機
                await new Promise(resolve => setTimeout(resolve, 10000));
                attempts++;
                
            } catch (error) {
                console.error(`Monitoring error: ${error.message}`);
                attempts++;
            }
        }
        
        throw new Error('Video generation timeout');
    }

    isRetryableError(error) {
        const retryableErrors = [
            'rate_limit_exceeded',
            'temporary_server_error',
            'network_timeout'
        ];
        
        return retryableErrors.some(err => 
            error.message.toLowerCase().includes(err)
        );
    }
}

// Make.com Webhook統合
class MakeWebhookHandler {
    constructor(runwayManager) {
        this.runwayManager = runwayManager;
    }

    async handleWebhook(webhookData) {
        try {
            // Google Sheetsからのデータ解析
            const request = this.parseSheetData(webhookData);
            
            // 動画生成処理
            const result = await this.runwayManager.processVideoRequest(request);
            
            // 結果をGoogle Sheetsに書き戻し
            await this.updateSheet(request.id, result);
            
            // 完了通知
            await this.sendNotification(request, result);
            
            return {
                status: 'success',
                message: 'Video processing completed',
                result: result
            };
            
        } catch (error) {
            console.error(`Webhook processing failed: ${error.message}`);
            return {
                status: 'error',
                message: error.message
            };
        }
    }

    parseSheetData(webhookData) {
        return {
            id: webhookData.row_id,
            projectId: webhookData.project_id,
            type: webhookData.video_type,
            prompt: webhookData.prompt,
            priority: webhookData.priority || 'normal',
            aspectRatio: webhookData.aspect_ratio || '16:9',
            platforms: webhookData.target_platforms?.split(',') || ['youtube']
        };
    }

    async updateSheet(requestId, result) {
        // Google Sheets API更新処理
        const updateData = {
            status: result.success ? 'completed' : 'failed',
            output_path: result.videoPath || '',
            generation_time: result.generationTime || 0,
            error_message: result.error || '',
            processed_at: new Date().toISOString()
        };

        // 実際のGoogle Sheets API呼び出し
        console.log(`Updating sheet for request ${requestId}:`, updateData);
    }

    async sendNotification(request, result) {
        if (result.success) {
            console.log(`✅ Video generated successfully for project ${request.projectId}`);
            // Slack/Discord/Email通知
        } else {
            console.log(`❌ Video generation failed for project ${request.projectId}: ${result.error}`);
            // エラー通知
        }
    }
}

// エクスポート
module.exports = {
    RunwayWorkflowManager,
    MakeWebhookHandler
};

// 使用例
/*
const config = {
    apiKey: process.env.RUNWAY_API_KEY
};

const runwayManager = new RunwayWorkflowManager(config);
const webhookHandler = new MakeWebhookHandler(runwayManager);

// Express.jsでの使用例
app.post('/webhook/runway', async (req, res) => {
    const result = await webhookHandler.handleWebhook(req.body);
    res.json(result);
});
*/
