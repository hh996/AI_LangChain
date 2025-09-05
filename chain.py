from operator import itemgetter

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
import os

from enum import Enum
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SerpAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 加载.env文件中的环境变量
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

class RetrievalMode(Enum):
    NO_FILE_RETRIEVER = 0
    WITH_FILE_RETRIEVER = 1


class SearchMode(Enum):
    LOCAL = 0
    WEB = 1



class RetrievalChain:
    def __init__(
            self,
            chat_model: str = "deepseek-chat",
            base_url: str = "https://api.deepseek.com"
    ):
        # 初始化网络搜索工具
        self.search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

        # 初始化模型
        self.model = ChatOpenAI(model=chat_model, base_url=base_url, api_key=OPENAI_API_KEY)
        self.output_parser = StrOutputParser()

        # 初始化提示模板
        # self.prompt = ChatPromptTemplate.from_messages([
        #     ("system", "你是一个基于知识库回答问题的助手。以下是相关文档内容：{context}"),
        #     ("human", "问题：{question}")
        # ])
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个AI助手"),
                ("human", "{context}"),
                MessagesPlaceholder(variable_name="history"),
            ]
        )
        # 对话缓存内存
        self.memory = ConversationBufferMemory(return_messages=True)

        # 初始化文本嵌入模型
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})

        self.chat_history = []
        self.search_mode = SearchMode.LOCAL
        self.retriever_mode = RetrievalMode.NO_FILE_RETRIEVER
        self.vectorstore = None
        self.chain = None

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def web_search(self, query: str) -> str:
        return self.search.run(query)

    def build_chain(self, runnable):
        self.chain = RunnableSequence(
            # {
            #     "context": runnable,
            #     "question": RunnablePassthrough()
            # },
            {
                "context": runnable,
                "history": RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
            },
            self.prompt,
            self.model,
            self.output_parser
        )

    def invoke(self, query: str) -> str:
        # 是否加载知识库
        if self.retriever_mode == RetrievalMode.WITH_FILE_RETRIEVER:
            runnable = self.vectorstore.as_retriever(search_kwargs={"k": 2}) | self.format_docs
        else:
            runnable = RunnablePassthrough()
        # 是否网络搜索
        if self.search_mode == SearchMode.WEB:
            runnable |= RunnableLambda(self.web_search)
        self.build_chain(runnable)
        # 执行链
        response = self.chain.invoke(query)
        self.memory.save_context({"input": query}, {"output": response})
        return response

# retrieval_chain = RetrievalChain()
# query = "pipenv AI"
# response = retrieval_chain.invoke(query)
# print(response)

