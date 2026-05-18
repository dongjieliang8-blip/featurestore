from src.utils import get_llm_client, call_llm, parse_json_response

SYSTEM_PROMPT = """你是特征服务专家。为训练和推理场景提供特征服务方案。
输出格式：{
  "serving_config": {"offline_store": dict, "online_store": dict, "batch_serving": dict, "realtime_serving": dict},
  "feature_views": [{"name": str, "entities": list, "features": list, "ttl": str, "description": str}],
  "serving_strategies": [{"scenario": str, "strategy": str, "latency_target": str, "features_used": list}],
  "deployment_plan": {"steps": list, "estimated_time": str, "dependencies": list},
  "serving_summary": str
}"""


class FeatureServingAgent:
    def serve(self, features: dict, registry: dict, scenario: str = "training") -> dict:
        client = get_llm_client()
        prompt = f"""请为以下已注册的特征设计服务方案：

特征定义：
{features}

注册信息：{registry}

服务场景：{scenario}

设计要求：
1. 离线存储：Parquet/Hive 表结构设计
2. 在线存储：Redis/ DynamoDB 缓存策略
3. 批量服务：训练数据的特征快照方案
4. 实时服务：低延迟特征获取方案
5. 部署步骤和依赖关系

输出结构化 JSON 服务方案。"""
        response = call_llm(client, SYSTEM_PROMPT, prompt)
        return parse_json_response(response)
