from autogen import AssistantAgent
def addition_tool(a: int, b: int) -> int:
            """计算两个数的和"""
            return a + b
class BasicAgent(AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interaction_count = 0
    
    def register_custom_tools(self):

        self.register_for_llm(
            name="addition_tool",
            description="计算两个数的和"
        )(addition_tool)

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
        system_message="你是一个可以计算加法的助手，当用户问加法问题时，使用工具回答。",
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
        human_input_mode="TERMINATE",  # 每次回复需用户输入
        is_termination_msg=lambda x: x.get("content", "") == "",
        # code_execution_config=False  # 不执行代码
    )
    user_proxy.register_for_execution(name="addition_tool")(addition_tool)
    # 启动对话：用户问一个加法问题
    user_proxy.initiate_chat(
        recipient=custom_agent,
        message=f"请计算{2**20}加{7**20}等于多少？"
    )