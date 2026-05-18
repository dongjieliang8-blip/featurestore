from src.utils import get_llm_client, call_llm, parse_json_response

SYSTEM_PROMPT = """你是特征工程专家。从原始数据中提取有意义的 ML 特征。
输出格式：{
  "features": [{"name": str, "type": str, "source_field": str, "description": str, "data_type": str, "importance": str}],
  "feature_groups": [{"name": str, "features": list, "description": str}],
  "raw_stats": {"row_count": int, "column_count": int, "missing_fields": list},
  "extraction_summary": str
}"""


class FeatureExtractorAgent:
    def extract(self, data: dict, task_description: str = "") -> dict:
        client = get_llm_client()
        prompt = f"""请从以下原始数据中提取 ML 特征：

数据结构：
{data}

任务描述：{task_description or "通用特征提取"}

分析要求：
1. 识别所有可用的数值型和类别型特征
2. 按业务含义对特征进行分组
3. 评估每个特征的潜在重要性（high/medium/low）
4. 识别数据质量问题（缺失值、异常值等）
5. 建议可能的特征工程操作（归一化、编码、交叉等）

输出结构化 JSON 特征提取报告。"""
        response = call_llm(client, SYSTEM_PROMPT, prompt)
        return parse_json_response(response)
