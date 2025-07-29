const { Client, GatewayIntentBits } = require('discord.js');
const axios = require('axios');

class MidjourneyBot {
    constructor(token, channelId) {
        this.token = token;
        this.channelId = channelId;
        this.client = new Client({
            intents: [
                GatewayIntentBits.Guilds,
                GatewayIntentBits.GuildMessages,
                GatewayIntentBits.MessageContent
            ]
        });
        
        this.setupEventHandlers();
    }

    setupEventHandlers() {
        this.client.once('ready', () => {
            console.log(`Logged in as ${this.client.user.tag}!`);
        });

        this.client.on('messageCreate', async (message) => {
            if (message.author.bot && message.author.username === 'Midjourney Bot') {
                await this.handleMidjourneyResponse(message);
            }
        });
    }

    async login() {
        await this.client.login(this.token);
    }

    async generateImage(prompt, options = {}) {
        const channel = this.client.channels.cache.get(this.channelId);
        if (!channel) {
            throw new Error('Channel not found');
        }

        const {
            aspectRatio = '1:1',
            quality = 2,
            stylize = 500,
            chaos = 0,
            seed = null
        } = options;

        let command = `/imagine prompt: ${prompt}`;
        command += ` --ar ${aspectRatio}`;
        command += ` --q ${quality}`;
        command += ` --stylize ${stylize}`;
        
        if (chaos > 0) {
            command += ` --chaos ${chaos}`;
        }
        
        if (seed) {
            command += ` --seed ${seed}`;
        }

        await channel.send(command);
        return this.waitForCompletion(prompt);
    }

    async waitForCompletion(prompt, timeout = 300000) {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                reject(new Error('Generation timeout'));
            }, timeout);

            const messageHandler = async (message) => {
                if (message.author.username === 'Midjourney Bot' && 
                    message.content.includes(prompt.substring(0, 50))) {
                    
                    if (message.attachments.size > 0) {
                        clearTimeout(timer);
                        this.client.off('messageCreate', messageHandler);
                        
                        const attachment = message.attachments.first();
                        resolve({
                            imageUrl: attachment.url,
                            messageId: message.id,
                            timestamp: new Date()
                        });
                    }
                }
            };

            this.client.on('messageCreate', messageHandler);
        });
    }

    async handleMidjourneyResponse(message) {
        // Webhook通知やデータベース保存などの後処理
        if (message.attachments.size > 0) {
            const attachment = message.attachments.first();
            await this.notifyCompletion({
                imageUrl: attachment.url,
                messageContent: message.content,
                timestamp: new Date()
            });
        }
    }

    async notifyCompletion(data) {
        // Zapier WebhookまたはMake.comへの通知
        try {
            await axios.post(process.env.WEBHOOK_URL, {
                type: 'midjourney_completion',
                data: data
            });
        } catch (error) {
            console.error('Notification failed:', error);
        }
    }
}

// 使用例
const bot = new MidjourneyBot(
    process.env.DISCORD_TOKEN,
    process.env.DISCORD_CHANNEL_ID
);

module.exports = MidjourneyBot;
