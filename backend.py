from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

prompt_template_file = 'prompts/prompt_template.md'
default_role_file = 'prompts/class_prompt.md'

with open(prompt_template_file) as f:
    prompt_template = f.read()
with open(default_role_file) as f:
    default_role = f.read()


class VIPSON:
    def __init__(self, api_key, role):
        self.llm = ChatOpenAI(model='gpt-4o-mini')
        self.system_prompt = SystemMessage(content=prompt_template.format(role=default_role) )
        self.history = []

    def step(self, user_input):
        self.history.append(HumanMessage(content=user_input))
        messages = [self.system_prompt] + self.history
        response = self.llm.invoke(messages)
        self.history.append(AIMessage(content=response.content))
        return response.content