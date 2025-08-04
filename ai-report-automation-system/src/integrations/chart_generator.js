/**
 * グラフ自動生成システム
 * Google Charts APIを使用した動的グラフ生成
 */

class ChartGenerator {
    constructor(config = {}) {
        this.config = {
            defaultWidth: config.width || 800,
            defaultHeight: config.height || 400,
            colorScheme: config.colorScheme || 'professional',
            ...config
        };
        
        this.colorSchemes = {
            professional: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            corporate: ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD'],
            modern: ['#3498DB', '#E67E22', '#2ECC71', '#E74C3C', '#9B59B6'],
            minimal: ['#34495E', '#7F8C8D', '#BDC3C7', '#95A5A6', '#AEB6BF']
        };
        
        this.loadGoogleCharts();
    }
    
    loadGoogleCharts() {
        return new Promise((resolve, reject) => {
            if (typeof google !== 'undefined' && google.charts) {
                resolve();
                return;
            }
            
            const script = document.createElement('script');
            script.src = 'https://www.gstatic.com/charts/loader.js';
            script.onload = () => {
                google.charts.load('current', {'packages': ['corechart', 'bar', 'line', 'gauge']});
                google.charts.setOnLoadCallback(resolve);
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }
    
    /**
     * 売上トレンドグラフ生成
     */
    generateSalesTrend(data, containerId, options = {}) {
        const chartData = this.prepareSalesTrendData(data);
        const chartOptions = {
            title: '売上トレンド',
            titleTextStyle: { fontSize: 18, color: '#333' },
            hAxis: {
                title: '期間',
                titleTextStyle: { color: '#666' },
                textStyle: { color: '#666' }
            },
            vAxis: {
                title: '売上金額 (万円)',
                titleTextStyle: { color: '#666' },
                textStyle: { color: '#666' },
                format: '#,###万円'
            },
            colors: this.colorSchemes[this.config.colorScheme],
            backgroundColor: 'transparent',
            chartArea: { width: '80%', height: '70%' },
            legend: { position: 'top', alignment: 'center' },
            curveType: 'function',
            lineWidth: 3,
            pointSize: 6,
            ...options
        };
        
        const chart = new google.visualization.LineChart(document.getElementById(containerId));
        chart.draw(chartData, chartOptions);
        
        return chart;
    }
    
    prepareSalesTrendData(rawData) {
        const data = new google.visualization.DataTable();
        data.addColumn('string', '期間');
        data.addColumn('number', '今期売上');
        data.addColumn('number', '前期売上');
        data.addColumn('number', '目標');
        
        rawData.periods.forEach(period => {
            data.addRows([[
                period.name,
                period.current || 0,
                period.previous || 0,
                period.target || 0
            ]]);
        });
        
        return data;
    }
    
    /**
     * 商品別売上グラフ生成
     */
    generateProductSales(data, containerId, options = {}) {
        const chartData = this.prepareProductSalesData(data);
        const chartOptions = {
            title: '商品別売上構成',
            titleTextStyle: { fontSize: 18, color: '#333' },
            is3D: false,
            colors: this.colorSchemes[this.config.colorScheme],
            backgroundColor: 'transparent',
            chartArea: { width: '90%', height: '80%' },
            legend: { 
                position: 'right', 
                alignment: 'center',
                textStyle: { fontSize: 12 }
            },
            pieSliceText: 'percentage',
            pieSliceTextStyle: { fontSize: 11 },
            tooltip: { 
                text: 'both',
                textStyle: { fontSize: 12 }
            },
            ...options
        };
        
        const chart = new google.visualization.PieChart(document.getElementById(containerId));
        chart.draw(chartData, chartOptions);
        
        return chart;
    }
    
    prepareProductSalesData(rawData) {
        const data = new google.visualization.DataTable();
        data.addColumn('string', '商品名');
        data.addColumn('number', '売上');
        
        rawData.products.forEach(product => {
            data.addRows([[product.name, product.sales]]);
        });
        
        return data;
    }
    
    /**
     * 顧客セグメント分析グラフ
     */
    generateCustomerSegment(data, containerId, options = {}) {
        const chartData = this.prepareCustomerSegmentData(data);
        const chartOptions = {
            title: '顧客セグメント分析',
            titleTextStyle: { fontSize: 18, color: '#333' },
            hAxis: {
                title: 'セグメント',
                titleTextStyle: { color: '#666' },
                textStyle: { color: '#666' }
            },
            vAxes: {
                0: {
                    title: '顧客数 (人)',
                    titleTextStyle: { color: this.colorSchemes[this.config.colorScheme][0] },
                    textStyle: { color: this.colorSchemes[this.config.colorScheme][0] }
                },
                1: {
                    title: '平均購入額 (円)',
                    titleTextStyle: { color: this.colorSchemes[this.config.colorScheme][1] },
                    textStyle: { color: this.colorSchemes[this.config.colorScheme][1] }
                }
            },
            series: {
                0: {
                    type: 'columns',
                    targetAxisIndex: 0,
                    color: this.colorSchemes[this.config.colorScheme][0]
                },
                1: {
                    type: 'line',
                    targetAxisIndex: 1,
                    color: this.colorSchemes[this.config.colorScheme][1],
                    lineWidth: 3,
                    pointSize: 8
                }
            },
            backgroundColor: 'transparent',
            chartArea: { width: '80%', height: '70%' },
            legend: { position: 'top', alignment: 'center' },
            ...options
        };
        
        const chart = new google.visualization.ComboChart(document.getElementById(containerId));
        chart.draw(chartData, chartOptions);
        
        return chart;
    }
    
    prepareCustomerSegmentData(rawData) {
        const data = new google.visualization.DataTable();
        data.addColumn('string', 'セグメント');
        data.addColumn('number', '顧客数');
        data.addColumn('number', '平均購入額');
        
        rawData.segments.forEach(segment => {
            data.addRows([[
                segment.name,
                segment.customerCount,
                segment.averagePurchase
            ]]);
        });
        
        return data;
    }
    
    /**
     * マーケティングROI分析グラフ
     */
    generateMarketingROI(data, containerId, options = {}) {
        const chartData = this.prepareMarketingROIData(data);
        const chartOptions = {
            title: 'マーケティングチャネル別ROI',
            titleTextStyle: { fontSize: 18, color: '#333' },
            hAxis: {
                title: 'チャネル',
                titleTextStyle: { color: '#666' },
                textStyle: { color: '#666' }
            },
            vAxis: {
                title: 'ROI (%)',
                titleTextStyle: { color: '#666' },
                textStyle: { color: '#666' },
                format: '#\'%\'',
                minValue: 0
            },
            colors: this.colorSchemes[this.config.colorScheme],
            backgroundColor: 'transparent',
            chartArea: { width: '80%', height: '70%' },
            legend: { position: 'none' },
            bar: { groupWidth: '75%' },
            ...options
        };
        
        const chart = new google.visualization.ColumnChart(document.getElementById(containerId));
        chart.draw(chartData, chartOptions);
        
        return chart;
    }
    
    prepareMarketingROIData(rawData) {
        const data = new google.visualization.DataTable();
        data.addColumn('string', 'チャネル');
        data.addColumn('number', 'ROI');
        
        rawData.channels.forEach(channel => {
            data.addRows([[channel.name, channel.roi]]);
        });
        
        return data;
    }
    
    /**
     * KPIダッシュボード生成
     */
    generateKPIDashboard(data, containerId, options = {}) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        
        // KPIカード作成
        data.kpis.forEach((kpi, index) => {
            const kpiCard = this.createKPICard(kpi, index);
            container.appendChild(kpiCard);
        });
        
        // ゲージチャート作成（重要KPIのみ）
        data.kpis.filter(kpi => kpi.showGauge).forEach((kpi, index) => {
            const gaugeContainer = document.createElement('div');
            gaugeContainer.id = `gauge-${index}`;
            gaugeContainer.style.cssText = 'width: 300px; height: 200px; display: inline-block; margin: 10px;';
            container.appendChild(gaugeContainer);
            
            this.generateGaugeChart(kpi, `gauge-${index}`);
        });
    }
    
    createKPICard(kpi, index) {
        const card = document.createElement('div');
        card.className = 'kpi-card';
        card.style.cssText = `
            display: inline-block;
            margin: 10px;
            padding: 20px;
            border-radius: 8px;
            background: linear-gradient(135deg, ${this.colorSchemes[this.config.colorScheme][index % 5]}, ${this.adjustColor(this.colorSchemes[this.config.colorScheme][index % 5], -20)});
            color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            min-width: 200px;
            text-align: center;
        `;
        
        const changeIcon = kpi.change >= 0 ? '↗' : '↘';
        const changeColor = kpi.change >= 0 ? '#4CAF50' : '#F44336';
        
        card.innerHTML = `
            <h3 style="margin: 0 0 10px 0; font-size: 16px;">${kpi.name}</h3>
            <div style="font-size: 28px; font-weight: bold; margin: 10px 0;">${kpi.value}</div>
            <div style="font-size: 14px; color: ${changeColor};">
                ${changeIcon} ${Math.abs(kpi.change)}% vs 前期
            </div>
        `;
        
        return card;
    }
    
    generateGaugeChart(kpi, containerId) {
        const data = google.visualization.arrayToDataTable([
            ['Label', 'Value'],
            [kpi.name, kpi.percentage || 0]
        ]);
        
        const options = {
            width: 300,
            height: 200,
            redFrom: 0,
            redTo: 30,
            yellowFrom: 30,
            yellowTo: 70,
            greenFrom: 70,
            greenTo: 100,
            minorTicks: 5,
            backgroundColor: 'transparent'
        };
        
        const chart = new google.visualization.Gauge(document.getElementById(containerId));
        chart.draw(data, options);
        
        return chart;
    }
    
    /**
     * 自動レイアウト生成
     */
    generateAutoLayout(analysisData, containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        
        // レスポンシブグリッドレイアウト
        container.style.cssText = `
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            padding: 20px;
        `;
        
        const charts = [];
        
        // 売上データがある場合
        if (analysisData.sales) {
            const salesDiv = this.createChartContainer('sales-trend', '売上トレンド');
            container.appendChild(salesDiv);
            charts.push(() => this.generateSalesTrend(analysisData.sales, 'sales-trend-chart'));
        }
        
        // 商品データがある場合
        if (analysisData.products) {
            const productDiv = this.createChartContainer('product-sales', '商品別売上');
            container.appendChild(productDiv);
            charts.push(() => this.generateProductSales(analysisData.products, 'product-sales-chart'));
        }
        
        // 顧客データがある場合
        if (analysisData.customers) {
            const customerDiv = this.createChartContainer('customer-segment', '顧客セグメント');
            container.appendChild(customerDiv);
            charts.push(() => this.generateCustomerSegment(analysisData.customers, 'customer-segment-chart'));
        }
        
        // マーケティングデータがある場合
        if (analysisData.marketing) {
            const marketingDiv = this.createChartContainer('marketing-roi', 'マーケティングROI');
            container.appendChild(marketingDiv);
            charts.push(() => this.generateMarketingROI(analysisData.marketing, 'marketing-roi-chart'));
        }
        
        // KPIデータがある場合
        if (analysisData.kpis) {
            const kpiDiv = this.createChartContainer('kpi-dashboard', 'KPIダッシュボード');
            kpiDiv.style.gridColumn = '1 / -1'; // 全幅使用
            container.appendChild(kpiDiv);
            charts.push(() => this.generateKPIDashboard(analysisData.kpis, 'kpi-dashboard-chart'));
        }
        
        // 全チャートを描画
        Promise.all([this.loadGoogleCharts()]).then(() => {
            charts.forEach(chartFunc => chartFunc());
        });
    }
    
    createChartContainer(id, title) {
        const container = document.createElement('div');
        container.style.cssText = `
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        `;
        
        const titleElement = document.createElement('h3');
        titleElement.textContent = title;
        titleElement.style.cssText = `
            margin: 0 0 20px 0;
            color: #333;
            font-size: 18px;
            border-bottom: 2px solid ${this.colorSchemes[this.config.colorScheme][0]};
            padding-bottom: 10px;
        `;
        
        const chartDiv = document.createElement('div');
        chartDiv.id = `${id}-chart`;
        chartDiv.style.cssText = `
            width: 100%;
            height: 400px;
        `;
        
        container.appendChild(titleElement);
        container.appendChild(chartDiv);
        
        return container;
    }
    
    /**
     * チャート画像出力
     */
    exportChartAsImage(chartElement, filename = 'chart.png') {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // SVGをCanvasに変換
        const svgData = new XMLSerializer().serializeToString(chartElement.querySelector('svg'));
        const img = new Image();
        
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            
            // 画像ダウンロード
            const link = document.createElement('a');
            link.download = filename;
            link.href = canvas.toDataURL();
            link.click();
        };
        
        img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
    }
    
    /**
     * 色調整ユーティリティ
     */
    adjustColor(color, amount) {
        const num = parseInt(color.replace("#", ""), 16);
        const amt = Math.round(2.55 * amount);
        const R = (num >> 16) + amt;
        const G = (num >> 8 & 0x00FF) + amt;
        const B = (num & 0x0000FF) + amt;
        
        return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
    }
}

/**
 * 使用例とサンプルデータ
 */
const sampleData = {
    sales: {
        periods: [
            { name: '1月', current: 1200, previous: 1000, target: 1100 },
            { name: '2月', current: 1350, previous: 1150, target: 1200 },
            { name: '3月', current: 1500, previous: 1300, target: 1400 }
        ]
    },
    products: {
        products: [
            { name: '商品A', sales: 450 },
            { name: '商品B', sales: 380 },
            { name: '商品C', sales: 320 },
            { name: '商品D', sales: 250 }
        ]
    },
    customers: {
        segments: [
            { name: 'プレミアム', customerCount: 150, averagePurchase: 25000 },
            { name: 'スタンダード', customerCount: 800, averagePurchase: 12000 },
            { name: 'ライト', customerCount: 1200, averagePurchase: 6000 }
        ]
    },
    marketing: {
        channels: [
            { name: 'Google広告', roi: 350 },
            { name: 'Facebook広告', roi: 280 },
            { name: 'メール', roi: 420 },
            { name: 'SEO', roi: 650 }
        ]
    },
    kpis: {
        kpis: [
            { name: '売上', value: '1,500万円', change: 15, percentage: 85, showGauge: true },
            { name: '新規顧客', value: '125人', change: -5, percentage: 75, showGauge: true },
            { name: 'CVR', value: '3.2%', change: 8, percentage: 90, showGauge: false },
            { name: 'CPA', value: '2,500円', change: -12, percentage: 78, showGauge: false }
        ]
    }
};

// 初期化と使用例
document.addEventListener('DOMContentLoaded', function() {
    const chartGenerator = new ChartGenerator({
        colorScheme: 'professional',
        width: 800,
        height: 400
    });
    
    // 自動レイアウト生成
    // chartGenerator.generateAutoLayout(sampleData, 'dashboard-container');
});
