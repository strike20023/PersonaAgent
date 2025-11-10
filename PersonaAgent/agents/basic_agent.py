from autogen import AssistantAgent

def weather_tool(city: str, date: str = "今天") -> str:
            """查询指定城市在指定日期的天气（模拟数据）。

            参数:
            - city: 城市名称，例如"北京"。
            - date: 日期描述，例如"今天"、"明天"，默认"今天"。

            返回:
            - 模拟的天气描述字符串。
            """
            mock = {
                "北京": {
                    "今天": "多云转晴，最高 26℃，最低 14℃，西北风 3 级，空气质量优",
                    "明天": "晴，最高 24℃，最低 13℃，北风 2 级，空气质量良"
                },
                "上海": {
                    "今天": "小雨，最高 23℃，最低 18℃，东南风 3 级，空气质量良",
                    "明天": "阴，最高 25℃，最低 19℃，东风 2 级，空气质量良"
                },
                "广州": {
                    "今天": "阵雨，最高 30℃，最低 24℃，南风 3-4 级，空气质量良",
                    "明天": "多云，最高 31℃，最低 25℃，南风 2-3 级，空气质量良"
                }
            }
            city_data = mock.get(city)
            if not city_data:
                return f"{date}{city}的天气数据暂不可用（模拟）。请更换城市试试。"
            return city_data.get(date, f"{date}{city}的天气数据暂不可用（模拟）。")
class BasicAgent(AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interaction_count = 0
    
    def register_custom_tools(self):
        # 注册天气查询模拟工具，供大模型通过函数调用使用
        self.register_for_llm(
            name="weather_tool",
            description="查询城市在特定日期的天气（返回模拟数据）"
        )(weather_tool)

    def generate_reply(self, messages, sender, **kwargs):
        # 每次生成回复前，交互次数+1
        self.interaction_count += 1
        print(f"===== 第 {self.interaction_count} 次交互 =====")

        # 调用父类方法生成默认回复（基于大模型）
        reply = super().generate_reply(messages, sender,** kwargs)
        return reply

if __name__ == "__main__":
    # 配置大模型（以OpenAI为例）
    llm_config = {
        "model": "qwen3-8b-mlx",
        "api_key": "your-openai-api-key",
        "base_url": "http://127.0.0.1:1234/v1",
        "price" : [0, 0],
        "temperature": 0.7
    }

    # 初始化自定义智能体
    custom_agent = BasicAgent(
        name="custom_assistant",
        system_message="你是一个城市天气查询助手。当用户提出天气相关问题时，调用 weather_tool 获取指定城市和日期的天气信息并作答。若问题未提供日期，默认查询‘今天’。",
        llm_config=llm_config,
        human_input_mode="NEVER"
    )
    # 注册自定义工具
    custom_agent.register_custom_tools()

    # 初始化用户代理（模拟用户）
    from autogen import UserProxyAgent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        system_message="用户",
        human_input_mode="NEVER",  # 每次回复需用户输入
        is_termination_msg=lambda x: x.get("content", "") == "",
        # code_execution_config=False  # 不执行代码
    )
    # 在用户代理侧注册工具的实际执行函数
    user_proxy.register_for_execution(name="weather_tool")(weather_tool)
    # 启动对话：用户问一个加法问题
    user_proxy.initiate_chat(
        recipient=custom_agent,
        message=f"今日北京天气如何？"
    )
    import json
    with open('chat_messages.json', 'w', encoding='utf-8') as f:
        json.dump(list(user_proxy.chat_messages.values())[0], f, ensure_ascii=False, indent=4)
        
