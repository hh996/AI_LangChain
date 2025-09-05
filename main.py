import os
import shutil
import warnings

import gradio
import gradio as gr
from langchain_community.vectorstores import Annoy
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chain import RetrievalChain, RetrievalMode
from fileload import FileLoader

warnings.filterwarnings("ignore")

def split_documents(documents, chunk_size=500, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print("文本分割结束")
    return text_splitter.split_documents(documents)

def upload_file(files: list[gradio.utils.NamedString]):
    # 如果当前目录下没有名为doc的文件夹，则创建一个
    if not os.path.exists("docs"):
        os.mkdir("docs")
    for up_loadfile in files:
        # file为包含路径的文件名， basename获取最后的文件名
        filename = os.path.basename(up_loadfile.name)
        dst_path = os.path.join("docs", filename)
        shutil.copy(up_loadfile.name, dst_path)  # 复制文件（保留原文件）

    loader = FileLoader(directory="docs")
    docs = loader.load()
    # 分割文本
    all_splits = split_documents(docs)

    retrieval_chain.vectorstore =  Annoy.from_documents(all_splits, retrieval_chain.embeddings, n_trees=50)
    retrieval_chain.retriever_mode = RetrievalMode.WITH_FILE_RETRIEVER
    print("upload file", retrieval_chain.retriever_mode)

def submit_message(message, use_web):
    # TODO 这里先实现无历史记录对话
    # if use_web == "使用":
    #     retrieval_chain.search_mode = SearchMode.WEB
    retrieval_chain.chat_history.append(gr.ChatMessage(role="user", content=message))
    retrieval_chain.chat_history.append(gr.ChatMessage(role="assistant", content=retrieval_chain.invoke(message)))
    return "", retrieval_chain.chat_history

def clear_messages():
    return "", []

with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>AI_LangChain</center></h1> <center><font size=3> </center></font>""")
    with gr.Row():
        with gr.Column(scale=1):
            # Radio 单选框
            use_web = gr.Radio(["使用", "不使用"], label="联网搜索", value="不使用", interactive=True)
            # File上传文件的组件
            file = gr.File(label="上传文件，仅支持文本文件", file_count="multiple", file_types=['.txt', '.md', '.docx', '.pdf'])
        with gr.Column(scale=4):
            with gr.Row():
                chatbot = gr.Chatbot(type="messages", height=500)
            with gr.Row():
                message = gr.Textbox(label="输入问题", placeholder="请输入问题", lines=1)
            with gr.Row():
                clear_history = gr.Button("🧹 清除历史对话")
                submit = gr.Button("🚀 提交")
        # 上传文件动作
        file.upload(fn=upload_file, inputs=file, outputs=None)
        # 提交按钮动作
        submit.click(fn=submit_message, inputs=[message, use_web], outputs=[message, chatbot])
        # 输入框回车
        message.submit(fn=submit_message, inputs=[message, use_web], outputs=[message, chatbot])
        # 清空历史对话动作
        clear_history.click(fn=clear_messages, inputs=[], outputs=[message, chatbot])



print("正在初始化检索链...")
try:
    retrieval_chain = RetrievalChain()
    print("检索链初始化完成")
except Exception as e:
    print(f"检索链初始化失败: {e}")
print("正在启动 Gradio 界面...")
# 记得关VPN
demo.launch()