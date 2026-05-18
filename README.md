# FeatureStore

多 Agent 协作 ML 特征工程流水线。

Multi-agent collaborative ML feature engineering pipeline.

## Architecture / 架构

```
Feature Extractor -> Feature Validator -> Feature Registry -> Feature Serving
       |                   |                   |                  |
       v                   v                   v                  v
    特征提取            特征验证            特征注册            特征服务
```

## Installation / 安装

```bash
pip install -r requirements.txt
cp .env.example .env  # Set API Key
```

## Usage / 使用

```bash
# Full pipeline / 完整流水线
python -m src.main extract --input ./demo/sample_data/customer_data.json

# Extract + Validate only / 仅提取和验证
python -m src.main validate --input ./demo/sample_data/customer_data.json

# Register features / 注册特征
python -m src.main register --input ./demo/sample_data/customer_data.json

# Design serving plan / 设计服务方案
python -m src.main serve --input featurestore_report.json
```

## Agent Responsibilities / Agent 职责

| Agent | Responsibility | 职责 |
|-------|----------------|------|
| **Feature Extractor** | Extract ML features from raw data | 从原始数据中提取 ML 特征 |
| **Feature Validator** | Validate feature quality and availability | 验证特征质量和可用性 |
| **Feature Registry** | Manage feature metadata and registration | 管理特征元数据和注册信息 |
| **Feature Serving** | Design serving plans for training/inference | 为训练和推理设计服务方案 |

## Core Features / 核心功能

- Automated feature extraction from structured data
- Multi-dimensional feature quality validation
- Feature versioning and lineage tracking
- Offline/online serving strategy design
- Batch and realtime serving support
- Feature schema management

## Tech Stack / 技术栈

- LLM: MiMo v2.5 (via OpenAI-compatible API)
- CLI: Click
- Output: Rich (tables, panels)
- Data: JSON input/output
