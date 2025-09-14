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

def retrieve_relevant_context(query, documents, embedding_model, index, top_k=3):
    query_vec = embedding_model.encode([query], convert_to_numpy=True)  # 将查询转为向量
    scores, indices = index.search(query_vec, top_k)

    context_parts = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        # 添加相似度分数信息
        similarity = 1 / (1 + score)  # 转换为相似度分数
        context_parts.append(
            f"[参考资料 {i + 1}] (相似度: {similarity:.3f})\n{documents[idx]}"
        )

    return "\n\n".join(context_parts)

class RetrievalMode(Enum):
    NO_FILE_RETRIEVER = 0   # 不使用知识库
    WITH_FILE_RETRIEVER = 1   # 使用知识库


class SearchMode(Enum):
    LOCAL = 0   # 本地问答
    WEB = 1   # 网络搜索



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
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个AI助手"),
                ("human", "{context}"),
                MessagesPlaceholder(variable_name="history"),
            ]
        )
        # 对话缓存内存
        self.memory = ConversationBufferMemory(return_messages=True)

        self.chat_history = []
        self.search_mode = SearchMode.LOCAL
        self.retriever_mode = RetrievalMode.NO_FILE_RETRIEVER
        self.chain = None

        self.docs = None
        self.emb_model = None
        self.index = None

    def web_search(self, query: str) -> str:
        return self.search.run(query)

    def build_chain(self, runnable):
        self.chain = RunnableSequence(
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
            context = retrieve_relevant_context(
                query, self.docs, self.emb_model, self.index, top_k=2
            )
            query = f"""你是一个智能助手，请基于以下资料和之前的对话历史，自然地回答用户的问题: {query}。\n{context}\n。"""
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

