from src.utils import get_llm_client, call_llm, parse_json_response

SYSTEM_PROMPT = """你是特征注册管理专家。管理特征的元数据和注册信息。
输出格式：{
  "registered_features": [{"name": str, "version": str, "status": str, "owner": str, "tags": list, "created_at": str, "schema": {"type": str, "shape": str, "dtype": str}}],
  "feature_lineage": [{"feature": str, "upstream": list, "transformations": list}],
  "metadata": {"total_features": int, "active": int, "deprecated": int, "draft": int},
  "registry_summary": str
}"""


class FeatureRegistryAgent:
    def register(self, features: dict, validation: dict) -> dict:
        client = get_llm_client()
        prompt = f"""请为以下已验证的特征生成注册信息：

特征定义：
{features}

验证结果：
{validation}

注册要求：
1. 为每个特征生成唯一名称和版本号
2. 分配特征状态（active/deprecated/draft）
3. 添加业务标签用于分类检索
4. 定义特征的 schema（类型、形状、数据类型）
5. 建立特征血缘关系（上游依赖和转换步骤）

输出结构化 JSON 注册报告。"""
        response = call_llm(client, SYSTEM_PROMPT, prompt)
        return parse_json_response(response)
