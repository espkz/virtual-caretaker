from langgraph.checkpoint.memory import MemorySaver
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

memory = MemorySaver()

class StoryChatbot():
    def __init__(self, model, api_key = None):
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.system_prompt = self.configure_system_prompt()
        self.llm = ChatOpenAI(model=model, api_key=api_key)

        self.prompt = self.build_prompt()

        self.history = []

        self.chain = self.build_chain()

    def build_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}"),
            ("system", "Relevant context:\n{context}")
        ])

    def build_chain(self):
        return (
                RunnableParallel(
                    {
                        "input": RunnablePassthrough(),
                        "history": lambda x: self.history,
                        "context": self.get_context
                    }
                )
                | self.prompt
                | self.llm
        )

    def set_character(self, new_role: str, new_age: int = None):
        self.role = new_role
        if new_age is not None:
            self.age = new_age

        # reset conversation history
        self.history = []

        # rebuild system prompt and template
        self.system_prompt = self.configure_system_prompt()
        self.prompt = self.build_prompt()
        self.chain = self.build_chain()

    def get_context(self, user_input):
        docs = self.retriever.get_relevant_documents(user_input["input"])
        return {"context" : "\n".join([d.page_content for d in docs])}


    def configure_system_prompt(self):
        with open("prompt.md", "r", encoding="utf-8") as f:
            prompt = f.read()
        prompt = prompt.format(role=self.role, story=self.story, age=self.age)
        return prompt


    def respond(self, user_input):
        result = self.chain.invoke({"input" : user_input})

        self.history.append(("user", user_input))
        self.history.append(("assistant", result.content))
        return result.content