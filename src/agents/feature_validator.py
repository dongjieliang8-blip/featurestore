from src.utils import get_llm_client, call_llm, parse_json_response

SYSTEM_PROMPT = """你是特征质量验证专家。验证 ML 特征的质量和可用性。
输出格式：{
  "validations": [{"feature_name": str, "status": str, "checks": [{"check": str, "passed": bool, "detail": str}], "score": int}],
  "quality_summary": {"total_features": int, "passed": int, "warnings": int, "failed": int},
  "recommendations": [{"feature": str, "action": str, "reason": str}],
  "overall_quality_score": int
}"""


class FeatureValidatorAgent:
    def validate(self, features: dict, data: dict) -> dict:
        client = get_llm_client()
        prompt = f"""请验证以下提取的特征的质量：

特征定义：
{features}

原始数据：
{data}

验证维度：
1. 数据完整性：缺失值比例、数据类型一致性
2. 特征分布：是否符合正态分布、是否存在极端偏斜
3. 特征相关性：特征间是否存在高度共线性
4. 业务合理性：特征值是否在合理范围内
5. 信息价值：特征是否具有足够的区分度

输出结构化 JSON 验证报告。"""
        response = call_llm(client, SYSTEM_PROMPT, prompt)
        return parse_json_response(response)
