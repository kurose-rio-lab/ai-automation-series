# 業界別カスタマイズガイド

AI自動レポート生成システムを各業界のニーズに合わせてカスタマイズするための包括的ガイド

## 目次
1. [製造業](#製造業)
2. [小売業・EC](#小売業ec)
3. [金融・保険業](#金融保険業)
4. [IT・ソフトウェア業](#itソフトウェア業)
5. [医療・ヘルスケア](#医療ヘルスケア)
6. [不動産業](#不動産業)
7. [教育業界](#教育業界)
8. [物流・運輸業](#物流運輸業)
9. [飲食・サービス業](#飲食サービス業)
10. [建設・インフラ業](#建設インフラ業)

---

## 製造業

### 業界特性
- 生産効率性と品質管理が最重要
- 在庫管理とサプライチェーン最適化
- 設備稼働率とメンテナンス計画
- 安全性とコンプライアンス重視

### カスタマイズ項目

#### データソース設定
```json
{
  "manufacturing_data_sources": {
    "production_data": {
      "source": "ERP_system",
      "metrics": ["production_volume", "cycle_time", "yield_rate", "downtime"],
      "update_frequency": "hourly"
    },
    "quality_data": {
      "source": "QC_system", 
      "metrics": ["defect_rate", "inspection_results", "rework_rate"],
      "update_frequency": "real_time"
    },
    "inventory_data": {
      "source": "WMS",
      "metrics": ["stock_levels", "turnover_rate", "shortage_alerts"],
      "update_frequency": "daily"
    },
    "equipment_data": {
      "source": "IoT_sensors",
      "metrics": ["OEE", "MTBF", "maintenance_schedule"],
      "update_frequency": "continuous"
    }
  }
}
分析プロンプト特化
製造業向け分析プロンプト:

あなたは製造業の生産管理専門家です。以下の生産データを分析し、
製造効率向上のための洞察を提供してください。

【分析データ】
- 生産量: {{production_volume}}
- 設備稼働率 (OEE): {{oee_rate}}
- 品質指標: {{quality_metrics}}
- 在庫状況: {{inventory_status}}

【重点分析項目】
1. 生産効率分析（OEE要因分析）
2. 品質トレンド分析（不良率、工程能力）
3. 在庫最適化（回転率、安全在庫）
4. 設備保全計画（予防保全、故障予測）
5. コスト削減機会の特定

【出力要求】
- 生産性向上の具体的提案
- 品質改善アクションプラン
- 在庫削減戦略
- 設備投資の優先順位
- リスク要因と対策
KPI設定
Copymanufacturing_kpis = {
    "production_efficiency": {
        "OEE": {"target": 85, "unit": "%"},
        "cycle_time": {"target": "reduce_10%", "unit": "minutes"},
        "throughput": {"target": "increase_15%", "unit": "units/hour"}
    },
    "quality_metrics": {
        "defect_rate": {"target": "<0.5", "unit": "%"},
        "first_pass_yield": {"target": ">98", "unit": "%"},
        "customer_complaints": {"target": "<5", "unit": "cases/month"}
    },
    "inventory_management": {
        "inventory_turnover": {"target": ">12", "unit": "times/year"},
        "stockout_rate": {"target": "<2", "unit": "%"},
        "carrying_cost": {"target": "reduce_20%", "unit": "% of sales"}
    },
    "maintenance": {
        "planned_maintenance_ratio": {"target": ">80", "unit": "%"},
        "MTBF": {"target": "increase_25%", "unit": "hours"},
        "maintenance_cost": {"target": "reduce_15%", "unit": "% of revenue"}
    }
}
レポートテンプレート
Copy{
  "manufacturing_report_template": {
    "sections": [
      {
        "title": "生産実績サマリー",
        "content": ["production_volume", "oee_trend", "quality_overview"]
      },
      {
        "title": "効率性分析",
        "content": ["bottleneck_analysis", "capacity_utilization", "improvement_opportunities"]
      },
      {
        "title": "品質管理状況",
        "content": ["quality_trends", "defect_analysis", "corrective_actions"]
      },
      {
        "title": "在庫・調達状況",
        "content": ["inventory_levels", "supplier_performance", "cost_analysis"]
      },
      {
        "title": "設備保全計画",
        "content": ["maintenance_schedule", "equipment_health", "investment_recommendations"]
      }
    ]
  }
}
小売業・EC
業界特性
売上・来客数の変動が大きい
季節性・トレンドの影響大
在庫回転と機会損失の最適化
顧客行動分析とパーソナライゼーション
カスタマイズ項目
データソース設定
Copy{
  "retail_data_sources": {
    "sales_data": {
      "source": "POS_system",
      "metrics": ["sales_amount", "transaction_count", "average_basket", "conversion_rate"],
      "segmentation": ["product_category", "store_location", "time_period"]
    },
    "customer_data": {
      "source": "CRM_system",
      "metrics": ["customer_count", "repeat_rate", "ltv", "acquisition_cost"],
      "behavioral_data": ["purchase_history", "browsing_pattern", "engagement_metrics"]
    },
    "inventory_data": {
      "source": "inventory_management",
      "metrics": ["stock_levels", "turnover_rate", "markdown_rate", "stockout_incidents"]
    },
    "marketing_data": {
      "source": "marketing_platforms",
      "metrics": ["campaign_performance", "channel_attribution", "customer_acquisition"]
    }
  }
}
小売業特化分析プロンプト
小売業向け分析プロンプト:

あなたは小売業の販売分析専門家です。以下の売上・顧客データを分析し、
売上拡大と収益性向上のための戦略的洞察を提供してください。

【分析データ】
- 売上実績: {{sales_performance}}
- 顧客行動: {{customer_behavior}}
- 商品動向: {{product_trends}}
- 在庫状況: {{inventory_status}}

【重点分析項目】
1. 売上トレンド分析（季節性、商品別、店舗別）
2. 顧客セグメント分析（RFM分析、行動パターン）
3. 商品パフォーマンス（ABC分析、クロスセル機会）
4. 在庫最適化（回転率、デッドストック対策）
5. マーケティング効果測定（ROI、チャネル貢献度）

【戦略提案要求】
- 売上拡大施策（商品、価格、プロモーション）
- 顧客体験向上策
- 在庫効率化計画
- マーケティング予算最適化
- 新規事業機会の提案
季節性・トレンド分析設定
Copyseasonal_analysis_config = {
    "seasonal_patterns": {
        "fashion_retail": {
            "spring_summer": {"months": [3,4,5,6,7,8], "trend_factors": ["weather", "events"]},
            "fall_winter": {"months": [9,10,11,12,1,2], "trend_factors": ["holidays", "temperature"]},
            "special_events": ["black_friday", "christmas", "new_year", "valentine"]
        },
        "food_retail": {
            "daily_patterns": {"peak_hours": [11,12,18,19], "low_hours": [14,15,16]},
            "weekly_patterns": {"peak_days": ["friday", "saturday"], "low_days": ["tuesday", "wednesday"]},
            "seasonal_products": ["summer_beverages", "winter_hot_foods", "holiday_specials"]
        }
    },
    "trend_indicators": {
        "social_media_buzz": {"weight": 0.3, "sources": ["twitter", "instagram", "tiktok"]},
        "search_volume": {"weight": 0.2, "source": "google_trends"},
        "competitor_analysis": {"weight": 0.2, "monitoring": ["pricing", "promotions", "inventory"]},
        "economic_indicators": {"weight": 0.3, "factors": ["consumer_confidence", "disposable_income"]}
    }
}
金融・保険業
業界特性
規制遵守とリスク管理が最重要
顧客の信用評価と資産管理
商品パフォーマンスと収益性分析
不正検知とセキュリティ
カスタマイズ項目
データソース設定
Copy{
  "financial_data_sources": {
    "portfolio_data": {
      "source": "portfolio_management_system",
      "metrics": ["asset_value", "returns", "risk_metrics", "allocation"],
      "compliance_tracking": true
    },
    "customer_data": {
      "source": "CRM_banking",
      "metrics": ["credit_score", "account_balance", "transaction_history", "product_usage"],
      "privacy_protection": "enhanced"
    },
    "risk_data": {
      "source": "risk_management",
      "metrics": ["var", "credit_risk", "market_risk", "operational_risk"],
      "real_time_monitoring": true
    },
    "regulatory_data": {
      "source": "compliance_system",
      "metrics": ["regulatory_capital", "liquidity_ratios", "compliance_violations"],
      "audit_trail": true
    }
  }
}
金融業特化分析プロンプト
金融業向けリスク・収益分析プロンプト:

あなたは金融業のリスク管理・投資分析専門家です。
規制要件を満たしながら収益最大化を図るための分析を行ってください。

【分析データ】
- ポートフォリオ状況: {{portfolio_performance}}
- リスク指標: {{risk_metrics}}
- 顧客属性: {{customer_segments}}
- 市場環境: {{market_conditions}}

【重点分析項目】
1. リスク・リターン分析（シャープレシオ、VaR、ストレステスト）
2. 信用リスク評価（デフォルト確率、信用格付け）
3. 流動性分析（資金調達コスト、流動性比率）
4. 収益性分析（ROE、ROA、手数料収入）
5. 規制対応状況（自己資本比率、準備金）

【戦略提案】
- ポートフォリオ最適化案
- リスク軽減策
- 新商品開発提案（規制準拠）
- 顧客獲得・維持戦略
- コンプライアンス強化策

【注意事項】
- 金融庁等の規制要件遵守
- 顧客情報保護の徹底
- リスク開示の適切性
コンプライアンス設定
Copycompliance_config = {
    "regulatory_frameworks": {
        "banking": ["basel_iii", "ffiec", "pci_dss"],
        "securities": ["mifid_ii", "sec_regulations"],
        "insurance": ["solvency_ii", "ifrs17"]
    },
    "data_protection": {
        "encryption": "aes_256",
        "access_control": "role_based",
        "audit_logging": "comprehensive",
        "data_retention": {
            "customer_data": "7_years",
            "transaction_data": "10_years",
            "compliance_records": "permanent"
        }
    },
    "risk_limits": {
        "credit_risk": {"single_exposure": "10%", "sector_concentration": "25%"},
        "market_risk": {"var_limit": "2%", "stress_test_frequency": "quarterly"},
        "operational_risk": {"loss_threshold": "1M", "incident_reporting": "immediate"}
    }
}
IT・ソフトウェア業
業界特性
プロジェクト管理とリソース配分
技術負債とシステム品質
顧客満足度とSLA管理
イノベーションと競合分析
カスタマイズ項目
データソース設定
Copy{
  "it_data_sources": {
    "project_data": {
      "source": "project_management_tools",
      "metrics": ["project_progress", "resource_utilization", "budget_variance", "milestone_achievement"],
      "tools_integration": ["jira", "azure_devops", "github"]
    },
    "system_performance": {
      "source": "monitoring_tools",
      "metrics": ["uptime", "response_time", "error_rate", "capacity_utilization"],
      "real_time_alerts": true
    },
    "customer_data": {
      "source": "support_systems",
      "metrics": ["ticket_volume", "resolution_time", "satisfaction_score", "churn_risk"],
      "integration": ["zendesk", "salesforce", "intercom"]
    },
    "development_metrics": {
      "source": "dev_tools",
      "metrics": ["code_quality", "deployment_frequency", "lead_time", "failure_rate"],
      "devops_integration": true
    }
  }
}
IT業界特化分析プロンプト
IT・ソフトウェア業向け分析プロンプト:

あなたはIT企業の技術・事業分析専門家です。
技術的負債管理と事業成長の両立を図るための分析を行ってください。

【分析データ】
- プロジェクト状況: {{project_status}}
- システム品質: {{system_metrics}}
- 顧客満足度: {{customer_satisfaction}}
- 開発効率: {{development_productivity}}

【技術分析項目】
1. システム健全性（稼働率、パフォーマンス、セキュリティ）
2. 開発生産性（コード品質、デプロイ頻度、リードタイム）
3. 技術負債評価（保守性、拡張性、リファクタリング優先度）
4. インフラ効率性（リソース使用率、コスト最適化）

【事業分析項目】
1. 顧客価値創出（機能利用率、満足度向上要因）
2. 市場ポジション（競合比較、技術優位性）
3. 成長機会（新技術導入、市場拡大可能性）
4. リスク評価（技術的リスク、市場リスク）

【提案要求】
- 技術投資の優先順位
- システム改善ロードマップ
- 開発プロセス最適化
- 顧客体験向上策
- イノベーション創出計画
DevOps・アジャイル指標
Copydevops_metrics = {
    "deployment_metrics": {
        "deployment_frequency": {"target": "daily", "current": "weekly"},
        "lead_time": {"target": "<1_day", "current": "3_days"},
        "mttr": {"target": "<1_hour", "current": "4_hours"},
        "change_failure_rate": {"target": "<5%", "current": "15%"}
    },
    "quality_metrics": {
        "code_coverage": {"target": ">80%", "minimum": "70%"},
        "technical_debt_ratio": {"target": "<10%", "current": "25%"},
        "security_vulnerabilities": {"target": "0_critical", "current": "3_critical"},
        "performance_benchmarks": {"response_time": "<200ms", "throughput": ">1000_rps"}
    },
    "team_productivity": {
        "velocity": {"target": "stable_growth", "tracking": "sprint_basis"},
        "sprint_completion": {"target": ">90%", "current": "75%"},
        "bug_leakage": {"target": "<5%", "current": "12%"},
        "customer_feedback_cycle": {"target": "<1_week", "current": "2_weeks"}
    }
}
医療・ヘルスケア
業界特性
患者安全と医療品質が最優先
厳格な規制遵守とプライバシー保護
診療効率と医療費最適化
エビデンスベースの意思決定
カスタマイズ項目
データソース設定（プライバシー準拠）
Copy{
  "healthcare_data_sources": {
    "clinical_data": {
      "source": "EMR_system",
      "metrics": ["patient_outcomes", "treatment_effectiveness", "readmission_rate", "length_of_stay"],
      "privacy_level": "hipaa_compliant",
      "anonymization": "required"
    },
    "operational_data": {
      "source": "hospital_management",
      "metrics": ["bed_occupancy", "staff_utilization", "resource_allocation", "cost_per_patient"],
      "real_time_monitoring": true
    },
    "quality_indicators": {
      "source": "quality_management",
      "metrics": ["infection_rates", "medication_errors", "patient_satisfaction", "mortality_rates"],
      "benchmarking": "national_standards"
    },
    "financial_data": {
      "source": "revenue_cycle",
      "metrics": ["revenue_per_patient", "collection_rate", "cost_reduction", "insurance_reimbursement"],
      "compliance_tracking": true
    }
  }
}
医療業界特化分析プロンプト
医療・ヘルスケア業界向け分析プロンプト:

あなたは医療データ分析と医療経営の専門家です。
患者安全を最優先に、医療品質向上と経営効率化の両立を図る分析を行ってください。

【分析データ】（患者情報は完全匿名化済み）
- 診療成果: {{clinical_outcomes}}
- 運営効率: {{operational_efficiency}}
- 品質指標: {{quality_metrics}}
- 医療経済: {{financial_performance}}

【重点分析項目】
1. 患者アウトカム分析（治療効果、安全性指標）
2. 医療品質評価（感染率、合併症率、満足度）
3. 運営効率分析（病床稼働率、スタッフ配置最適化）
4. 医療経済効果（コスト削減、収益改善）
5. リスク管理（医療安全、コンプライアンス）

【エビデンス要求】
- 統計的有意性の確認
- 臨床ガイドライン準拠性
- ベンチマーク比較（全国平均、同規模施設）
- リスク・ベネフィット評価

【提案制約】
- 患者安全を最優先
- 医療倫理の遵守
- 規制要件の完全準拠
- エビデンスベースの推奨のみ
HIPAA準拠設定
Copyhipaa_compliance_config = {
    "data_protection": {
        "encryption": {
            "at_rest": "aes_256",
            "in_transit": "tls_1.3",
            "key_management": "hsm_based"
        },
        "access_control": {
            "authentication": "multi_factor",
            "authorization": "role_based_granular",
            "audit_logging": "comprehensive"
        },
        "data_minimization": {
            "collect_only_necessary": true,
            "retention_period": "as_required_by_law",
            "automatic_deletion": true
        }
    },
    "privacy_safeguards": {
        "de_identification": {
            "method": "safe_harbor_plus_expert_determination",
            "verification": "statistical_analysis",
            "re_identification_risk": "<0.09%"
        },
        "consent_management": {
            "explicit_consent": true,
            "opt_out_mechanism": true,
            "consent_tracking": "complete_audit_trail"
        }
    },
    "breach_prevention": {
        "monitoring": "real_time_anomaly_detection",
        "incident_response": "automated_containment",
        "notification": "within_60_days",
        "documentation": "comprehensive_reporting"
    }
}
不動産業
業界特性
市場動向と価格変動の分析
物件パフォーマンスと収益性
顧客ニーズとマッチング精度
地域特性と立地評価
カスタマイズ項目
データソース設定
Copy{
  "real_estate_data_sources": {
    "market_data": {
      "source": "market_intelligence",
      "metrics": ["property_prices", "market_volume", "price_trends", "inventory_levels"],
      "geographic_segmentation": ["prefecture", "city", "district", "neighborhood"]
    },
    "property_data": {
      "source": "property_management",
      "metrics": ["occupancy_rate", "rental_yield", "property_value", "maintenance_cost"],
      "property_types": ["residential", "commercial", "industrial", "retail"]
    },
    "customer_data": {
      "source": "crm_real_estate",
      "metrics": ["inquiry_volume", "conversion_rate", "customer_satisfaction", "repeat_business"],
      "customer_segments": ["buyers", "sellers", "renters", "investors"]
    },
    "economic_indicators": {
      "source": "economic_data_feeds",
      "metrics": ["interest_rates", "population_growth", "employment_rate", "infrastructure_development"],
      "macro_factors": true
    }
  }
}
不動産業特化分析プロンプト
不動産業向け市場・投資分析プロンプト:

あなたは不動産市場分析と投資評価の専門家です。
市場動向を踏まえた投資判断と事業戦略の提案を行ってください。

【分析データ】
- 市場動向: {{market_trends}}
- 物件パフォーマンス: {{property_performance}}
- 顧客動向: {{customer_behavior}}
- 経済指標: {{economic_indicators}}

【重点分析項目】
1. 市場トレンド分析（価格動向、需給バランス、将来予測）
2. 立地評価（交通利便性、周辺環境、開発計画影響）
3. 投資収益性（利回り、キャッシュフロー、資産価値変動）
4. リスク評価（市場リスク、流動性リスク、規制リスク）
5. 顧客ニーズ分析（購買行動、価格感応度、優先条件）

【地域特性考慮】
- 人口動態変化の影響
- 交通インフラ整備効果
- 商業施設・教育機関の充実度
- 災害リスクと保険コスト
- 自治体政策の影響

【投資判断支援】
- 物件別投資適格性評価
- ポートフォリオ最適化提案
- 市場タイミング分析
- リスク分散戦略
- 出口戦略の検討
立地評価アルゴリズム
Copylocation_scoring_algorithm = {
    "transportation_score": {
        "weight": 0.25,
        "factors": {
            "station_distance": {"excellent": "<300m", "good": "<500m", "fair": "<800m"},
            "transportation_lines": {"multiple_lines": 10, "single_line": 5, "bus_only": 2},
            "accessibility": {"express_stop": 10, "local_stop": 7, "transfer_required": 4}
        }
    },
    "amenity_score": {
        "weight": 0.20,
        "factors": {
            "shopping": {"department_store": 10, "supermarket": 7, "convenience": 5},
            "education": {"prestigious_school": 10, "good_school": 7, "average_school": 5},
            "medical": {"hospital": 8, "clinic": 5, "pharmacy": 3},
            "parks": {"large_park": 8, "small_park": 5, "playground": 3}
        }
    },
    "economic_potential": {
        "weight": 0.20,
        "factors": {
            "development_plans": {"major_development": 10, "minor_development": 5, "none": 0},
            "population_growth": {"growing": 8, "stable": 5, "declining": 2},
            "employment_opportunities": {"business_district": 10, "mixed_use": 7, "residential": 4}
        }
    },
    "risk_factors": {
        "weight": 0.15,
        "factors": {
            "disaster_risk": {"low": 10, "medium": 6, "high": 2},
            "crime_rate": {"very_low": 10, "low": 8, "average": 5, "high": 2},
            "environmental": {"excellent": 10, "good": 7, "average": 5, "poor": 2}
        }
    },
    "market_dynamics": {
        "weight": 0.20,
        "factors": {
            "price_trend": {"rising": 8, "stable": 6, "declining": 3},
            "liquidity": {"high": 10, "medium": 7, "low": 4},
            "rental_demand": {"high": 10, "medium": 7, "low": 4}
        }
    }
}
教育業界
業界特性
学習成果と教育品質の向上
学生・生徒の満足度と定着率
教職員の効率性と成長支援
経営効率化と持続可能性
カスタマイズ項目
データソース設定
Copy{
  "education_data_sources": {
    "academic_data": {
      "source": "LMS_system",
      "metrics": ["student_performance", "completion_rates", "engagement_metrics", "assessment_scores"],
      "privacy_protection": "ferpa_compliant"
    },
    "operational_data": {
      "source": "school_management",
      "metrics": ["enrollment_numbers", "retention_rate", "class_utilization", "resource_allocation"],
      "institutional_metrics": true
    },
    "financial_data": {
      "source": "financial_system",
      "metrics": ["tuition_revenue", "cost_per_student", "financial_aid", "operational_efficiency"],
      "budget_tracking": true
    },
    "satisfaction_data": {
      "source": "survey_systems",
      "metrics": ["student_satisfaction", "parent_feedback", "teacher_evaluation", "alumni_engagement"],
      "feedback_analysis": true
    }
  }
}
教育業界特化分析プロンプト
教育業界向け学習成果・運営分析プロンプト:

あなたは教育データ分析と学校経営の専門家です。
学習成果向上と持続可能な教育機関運営の両立を図る分析を行ってください。

【分析データ】（学生情報は完全匿名化済み）
- 学習成果: {{academic_performance}}
- 学生エンゲージメント: {{student_engagement}}
- 運営効率: {{operational_metrics}}
- 満足度指標: {{satisfaction_scores}}

【教育効果分析】
1. 学習成果評価（成績向上、スキル習得、進路実現）
2. 教育方法効果（授業形式、教材、評価方法の影響）
3. 学生サポート効果（指導体制、相談体制、課外活動）
4. テクノロジー活用効果（デジタル教材、オンライン学習）

【運営効率分析】
1. リソース配分最適化（教職員配置、施設利用、予算配分）
2. 学生募集・定着戦略（入学者数、中退率、満足度向上）
3. 教職員開発（研修効果、モチベーション、離職率対策）
4. 財務健全性（収支バランス、コスト効率、投資効果）

【改善提案要求】
- 教育品質向上施策
- 学生満足度向上策
- 運営効率化計画
- 教職員育成プログラム
- 持続可能な財務戦略

【制約条件】
- 教育の公共性重視
- 学生プライバシー保護
- 教育理念との整合性
- 長期的視点での判断
学習分析ダッシュボード設定
Copylearning_analytics_config = {
    "student_success_metrics": {
        "academic_performance": {
            "gpa_tracking": {"threshold_risk": 2.5, "intervention_trigger": 2.0},
            "course_completion": {"target": ">90%", "at_risk": "<80%"},
            "skill_progression": {"competency_based": true, "mastery_threshold": 80}
        },
        "engagement_indicators": {
            "lms_activity": {"login_frequency": "daily", "content_interaction": "active"},
            "class_participation": {"attendance": ">90%", "participation_score": ">7/10"},
            "assignment_submission": {"on_time_rate": ">85%", "quality_score": ">80%"}
        },
        "early_warning_system": {
            "risk_factors": ["low_attendance", "poor_grades", "low_engagement", "financial_stress"],
            "intervention_protocols": ["academic_counseling", "tutoring", "financial_aid", "mental_health_support"],
            "success_tracking": "outcome_measurement"
        }
    },
    "institutional_effectiveness": {
        "retention_rates": {
            "first_year": {"target": ">85%", "benchmark": "national_average"},
            "overall": {"target": ">80%", "trend_monitoring": true}
        },
        "graduation_outcomes": {
            "completion_rate": {"target": ">70%", "time_to_graduation": "track_progress"},
            "employment_rate": {"within_6months": ">80%", "field_relevance": ">70%"},
            "alumni_satisfaction": {"career_preparation": ">4.0/5.0", "overall_experience": ">4.2/5.0"}
        }
    }
}
物流・運輸業
業界特性
配送効率と時間管理
コスト最適化と収益性
顧客満足度と配送品質
環境対応と持続可能性
カスタマイズ項目
データソース設定
Copy{
  "logistics_data_sources": {
    "delivery_data": {
      "source": "TMS_system",
      "metrics": ["delivery_time", "on_time_rate", "route_efficiency", "fuel_consumption"],
      "real_time_tracking": true
    },
    "warehouse_data": {
      "source": "WMS_system", 
      "metrics": ["inventory_turnover", "picking_accuracy", "storage_utilization", "processing_time"],
      "automation_metrics": true
    },
    "cost_data": {
      "source": "financial_system",
      "metrics": ["cost_per_delivery", "fuel_costs", "labor_costs", "vehicle_maintenance"],
      "profitability_analysis": true
    },
    "customer_data": {
      "source": "customer_portal",
      "metrics": ["delivery_satisfaction", "complaint_rate", "repeat_business", "service_ratings"],
      "feedback_integration": true
    }
  }
}
物流業特化分析プロンプト
物流・運輸業向け効率・最適化分析プロンプト:

あなたは物流オペレーションと運輸効率化の専門家です。
配送品質を維持しながらコスト効率を最大化する分析を行ってください。

【分析データ】
- 配送実績: {{delivery_performance}}
- 倉庫運営: {{warehouse_operations}}
- コスト構造: {{cost_breakdown}}
- 顧客満足度: {{customer_satisfaction}}

【効率性分析項目】
1. 配送効率（時間短縮、ルート最適化、積載率向上）
2. 倉庫効率（在庫回転、ピッキング精度、保管効率）
3. コスト分析（配送単価、燃料効率、人件費最適化）
4. 品質管理（配送品質、破損率、遅延率改善）
5. 技術活用（IoT、AI、自動化の効果測定）

【持続可能性分析】
1. 環境負荷削減（CO2排出量、燃料効率、電動化）
2. 働き方改善（労働時間、安全性、職場環境）
3. 地域貢献（雇用創出、地域密着サービス）

【最適化提案】
- ルート最適化計画
- 在庫配置戦略
- 自動化投資優先順位
- 燃料コスト削減策
- 顧客満足度向上施策
- ESG対応計画

【KPI設定】
- 配送時間短縮目標
- コスト削減目標
- 品質向上目標
- 環境負荷削減目標
ルート最適化アルゴリズム
Copyroute_optimization_config = {
    "optimization_objectives": {
        "primary": "minimize_total_cost",
        "secondary": ["maximize_on_time_delivery", "minimize_fuel_consumption", "balance_driver_workload"],
        "constraints": ["vehicle_capacity", "time_windows", "driver_hours", "traffic_conditions"]
    },
    "cost_factors": {
        "fuel_cost": {"weight": 0.4, "price_per_liter": 150, "consumption_rate": "vehicle_specific"},
        "labor_cost": {"weight": 0.3, "hourly_rate": 2000, "overtime_multiplier": 1.25},
        "vehicle_cost": {"weight": 0.2, "depreciation": "time_distance_based", "maintenance": "mileage_based"},
        "penalty_cost": {"weight": 0.1, "late_delivery": 5000, "customer_complaint": 10000}
    },
    "optimization_algorithms": {
        "short_term": {"method": "genetic_algorithm", "real_time_updates": true},
        "medium_term": {"method": "simulated_annealing", "demand_forecasting": true},
        "long_term": {"method": "machine_learning", "pattern_recognition": true}
    },
    "performance_metrics": {
        "efficiency": ["distance_reduction", "time_reduction", "fuel_savings"],
        "quality": ["on_time_rate", "customer_satisfaction", "delivery_accuracy"],
        "sustainability": ["co2_reduction", "vehicle_utilization", "empty_miles"]
    }
}
飲食・サービス業
業界特性
顧客体験と満足度重視
食材管理と原価管理
スタッフ管理とサービス品質
季節性と地域性への対応
カスタマイズ項目
データソース設定
Copy{
  "restaurant_data_sources": {
    "sales_data": {
      "source": "POS_system",
      "metrics": ["daily_sales", "average_ticket", "table_turnover", "peak_hour_analysis"],
      "menu_performance": true
    },
    "customer_data": {
      "source": "reservation_system",
      "metrics": ["customer_frequency", "satisfaction_scores", "review_ratings", "loyalty_metrics"],
      "feedback_integration": true
    },
    "inventory_data": {
      "source": "inventory_management",
      "metrics": ["food_cost_percentage", "waste_rate", "stock_turnover", "supplier_performance"],
      "freshness_tracking": true
    },
    "staff_data": {
      "source": "HR_system",
      "metrics": ["staff_productivity", "service_quality", "training_effectiveness", "retention_rate"],
      "scheduling_optimization": true
    }
  }
}
飲食業特化分析プロンプト
飲食・サービス業向け顧客満足・収益分析プロンプト:

あなたは飲食店経営とホスピタリティの専門家です。
顧客満足度向上と収益性改善を両立する分析を行ってください。

【分析データ】
- 売上実績: {{sales_performance}}
- 顧客動向: {{customer_behavior}}
- 原価管理: {{cost_management}}
- サービス品質: {{service_quality}}

【顧客体験分析】
1. 顧客満足度要因（料理品質、サービス、雰囲気、価格満足度）
2. リピート行動分析（来店頻度、推奨意向、口コミ効果）
3. 客層分析（年代別、用途別、時間帯別の特徴）
4. 競合比較（差別化要因、競争優位性）

【収益性分析】
1. メニュー収益性（人気度vs利益率、ABC分析）
2. 原価管理（食材コスト、廃棄ロス、仕入れ最適化）
3. 人件費効率（時間帯別配置、生産性、研修効果）
4. 固定費配分（家賃、光熱費、設備償却の効率性）

【運営改善提案】
- メニュー構成最適化
- 価格戦略の見直し
- サービス品質向上策
- スタッフ教育プログラム
- 集客・リピート促進策
- コスト削減計画

【季節・地域特性】
- 季節メニューの効果
- 地域イベント連動施策
- 天候・曜日パターン分析
- 商圏内競合状況
メニュー最適化分析
Copymenu_optimization_config = {
    "menu_analysis_framework": {
        "profitability_matrix": {
            "stars": {"popularity": "high", "profit_margin": "high", "strategy": "promote_heavily"},
            "plowhorses": {"popularity": "high", "profit_margin": "low", "strategy": "cost_reduction"},
            "puzzles": {"popularity": "low", "profit_margin": "high", "strategy": "marketing_boost"},
            "dogs": {"popularity": "low", "profit_margin": "low", "strategy": "consider_removal"}
        },
        "cost_analysis": {
            "food_cost_percentage": {"target": "<30%", "acceptable": "<35%", "critical": ">40%"},
            "labor_cost_factor": {"prep_time": "minutes", "skill_level": "1-5", "equipment_use": "complexity"},
            "waste_factor": {"perishability": "shelf_life", "portion_control": "standardization"}
        },
        "customer_preference": {
            "satisfaction_scores": {"rating_threshold": 4.0, "review_sentiment": "positive"},
            "order_frequency": {"repeat_orders": "customer_loyalty", "seasonal_trends": "timing"},
            "demographic_appeal": {"age_groups": "segmentation", "dietary_preferences": "accommodation"}
        }
    },
    "pricing_strategy": {
        "psychological_pricing": {"charm_pricing": true, "anchor_pricing": "premium_items"},
        "competitive_pricing": {"market_position": "value_vs_premium", "price_sensitivity": "elasticity"},
        "dynamic_pricing": {"time_based": "happy_hour", "demand_based": "peak_times", "inventory_based": "perishables"}
    }
}
建設・インフラ業
業界特性
プロジェクト管理と工期遵守
安全管理と品質保証
資材管理とコスト管理
規制遵守と環境配慮
カスタマイズ項目
データソース設定
Copy{
  "construction_data_sources": {
    "project_data": {
      "source": "project_management",
      "metrics": ["project_progress", "milestone_achievement", "budget_variance", "schedule_adherence"],
      "gantt_integration": true
    },
    "safety_data": {
      "source": "safety_management",
      "metrics": ["accident_rate", "near_miss_incidents", "safety_training_compliance", "inspection_results"],
      "real_time_monitoring": true
    },
    "resource_data": {
      "source": "resource_management",
      "metrics": ["material_consumption", "equipment_utilization", "labor_productivity", "supplier_performance"],
      "supply_chain_tracking": true
    },
    "quality_data": {
      "source": "quality_control",
      "metrics": ["inspection_pass_rate", "rework_percentage", "defect_density", "customer_acceptance"],
      "compliance_tracking": true
    }
  }
}
建設業特化分析プロンプト
建設・インフラ業向けプロジェクト・安全分析プロンプト:

あなたは建設プロジェクト管理と安全管理の専門家です。
安全性を最優先に、品質・コスト・工期の最適化を図る分析を行ってください。

【分析データ】
- プロジェクト進捗: {{project_status}}
- 安全管理状況: {{safety_metrics}}
- 資源活用状況: {{resource_utilization}}
- 品質管理状況: {{quality_control}}

【安全性分析（最優先）】
1. 安全指標評価（事故率、ヒヤリハット、安全教育効果）
2. リスク要因分析（作業環境、気象条件、設備状態）
3. 安全対策効果（保護具使用、安全手順遵守、改善施策）
4. 法規制遵守状況（労働安全衛生法、建設業法）

【プロジェクト管理分析】
1. 進捗管理（工程遅延要因、クリティカルパス分析）
2. コスト管理（予算統制、原価分析、利益率改善）
3. 品質管理（検査結果、手直し率、顧客満足度）
4. リソース効率（人員配置、機械稼働率、材料調達）

【改善提案要求】
- 安全性向上の最優先施策
- 工期短縮可能性（安全性維持前提）
- コスト削減機会
- 品質向上策
- 生産性改善計画
- 環境負荷削減策

【制約条件】
- 安全基準の厳格遵守
- 建築基準法等の法規制遵守
- 品質基準の維持
- 環境規制の遵守
安全管理KPI設定
Copysafety_management_kpis = {
    "leading_indicators": {
        "safety_training": {
            "completion_rate": {"target": "100%", "tracking": "monthly"},
            "certification_currency": {"target": "100%", "renewal_tracking": true},
            "toolbox_meetings": {"frequency": "daily", "attendance": ">95%"}
        },
        "safety_inspections": {
            "inspection_frequency": "weekly", 
            "checklist_completion": "100%",
            "corrective_action_completion": {"target": "within_24hours", "tracking": true}
        },
        "risk_assessments": {
            "jsa_completion": {"target": "100%", "before_work": true},
            "hazard_identification": {"reporting_rate": "encourage", "follow_up": "mandatory"},
            "risk_mitigation": {"implementation_rate": "100%", "effectiveness_review": "monthly"}
        }
    },
    "lagging_indicators": {
        "incident_rates": {
            "ltir": {"target": "0", "industry_benchmark": "compare"},
            "trir": {"target": "<1.0", "trend_monitoring": true},
            "near_miss_ratio": {"target": "10:1", "reporting_culture": "positive"}
        },
        "severity_measures": {
            "lost_workdays": {"target": "minimize", "root_cause_analysis": "mandatory"},
            "medical_treatment_cases": {"trend": "decreasing", "prevention_focus": true},
            "property_damage": {"target": "minimize", "cost_tracking": true}
        }
    },
    "compliance_metrics": {
        "regulatory_compliance": {
            "osha_citations": {"target": "0", "prevention": "proactive_audits"},
            "permit_compliance": {"rate": "100%", "tracking": "real_time"},
            "environmental_compliance": {"violations": "0", "monitoring": "continuous"}
        }
    }
}
共通実装ガイド
業界共通カスタマイズ手順
1. 初期設定
Copy# 1. 設定ファイルのコピー
cp config/default_config.json config/industry_config.json

# 2. 業界特化データソースの設定
# industry_config.json を編集

# 3. 分析プロンプトのカスタマイズ
cp templates/default_prompts.txt templates/industry_prompts.txt
# industry_prompts.txt を編集

# 4. KPI設定の更新
# kpi_definitions.json を業界要件に合わせて更新
2. データソース連携
Copy# industry_data_connector.py
class IndustryDataConnector:
    def __init__(self, industry_type, config):
        self.industry_type = industry_type
        self.config = config
        self.data_sources = self._load_industry_sources()
    
    def _load_industry_sources(self):
        industry_config = f"config/{self.industry_type}_config.json"
        with open(industry_config, 'r') as f:
            return json.load(f)
    
    def connect_data_sources(self):
        for source_name, source_config in self.data_sources.items():
            connector = self._create_connector(source_config)
            yield source_name, connector
3. 業界特化分析の実装
Copy# industry_analyzer.py
class IndustryAnalyzer:
    def __init__(self, industry_type):
        self.industry_type = industry_type
        self.prompts = self._load_industry_prompts()
        self.kpis = self._load_industry_kpis()
    
    def analyze(self, data):
        # 業界特化の分析ロジック
        analysis_prompt = self.prompts[self.industry_type]
        results = self._run_analysis(data, analysis_prompt)
        return self._format_results(results)
4. 業界コンプライアンス対応
Copy# compliance_manager.py
class ComplianceManager:
    def __init__(self, industry_type):
        self.industry_type = industry_type
        self.regulations = self._load_regulations()
    
    def validate_compliance(self, data, analysis_results):
        # 業界規制への準拠チェック
        for regulation in self.regulations[self.industry_type]:
            compliance_result = self._check_regulation(data, regulation)
            if not compliance_result.compliant:
                self._handle_non_compliance(compliance_result)
導入後の最適化
パフォーマンス監視
分析精度の継続的評価
ユーザーフィードバックの収集
システムパフォーマンスの監視
継続的改善
業界トレンドへの対応
新しい規制要件への対応
分析アルゴリズムの改善
サポート体制
業界専門家との連携
ユーザートレーニングの実施
技術サポートの提供
