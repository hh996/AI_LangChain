import os
import warnings

import faiss
import fitz
import gradio
import gradio as gr
from docx import Document
from langchain_community.vectorstores import Annoy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from chain import RetrievalChain, RetrievalMode, SearchMode
from fileload import FileLoader

warnings.filterwarnings("ignore")

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == ".pdf":
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        elif ext == ".docx":
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise e
    return text

def split_text(text, chunk_size=256, overlap=32):
    """相邻块之间有32个词的重叠，避免关键信息被分割"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start >= len(words):
            break
    return chunks



def upload_file(files: list[gradio.utils.NamedString]):
    try:
        if not files:
            return "❌ 请选择要上传的文件"

        documents = []
        supported_exts = {".txt", ".pdf", ".docx"}
        processed_files = []

        # 处理上传的文件
        for uploaded_file in files:
            filename = os.path.basename(uploaded_file.name)
            file_ext = os.path.splitext(filename)[1].lower()
            
            # 检查文件格式
            if file_ext not in supported_exts:
                continue
                
            try:
                # 直接从上传的文件路径读取内容
                text = load_document(uploaded_file.name)
                if text.strip():
                    chunks = split_text(text, chunk_size=256, overlap=32)
                    chunks_with_source = [f"[来源: {filename}]\n{chunk}" for chunk in chunks]
                    documents.extend(chunks_with_source)
                    processed_files.append(filename)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                continue

        if not documents:
            return "❌ 没有找到有效的文档内容或文件格式不支持（仅支持 .txt, .pdf, .docx）"

        # 构建向量数据库
        print("开始构建向量数据库...")
        embedding_model_name = "all-MiniLM-L6-v2"
        embedding_model = SentenceTransformer(embedding_model_name)
        embeddings = embedding_model.encode(
            documents, convert_to_numpy=True, show_progress_bar=True
        )
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # 更新检索链
        retrieval_chain.docs = documents
        retrieval_chain.emb_model = embedding_model
        retrieval_chain.index = index
        retrieval_chain.retriever_mode = RetrievalMode.WITH_FILE_RETRIEVER
        
        print("向量数据库构建完成!")
        return f"✅ 成功处理了 {len(processed_files)} 个文件，构建了包含 {len(documents)} 个文档块的向量数据库\n处理的文件: {', '.join(processed_files)}"
        
    except Exception as e:
        error_msg = f"❌ 文件处理失败: {str(e)}"
        print(error_msg)
        return error_msg

def submit_message(message, use_web):
    if use_web == "使用":
        retrieval_chain.search_mode = SearchMode.WEB
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
            # 状态显示
            status = gr.Textbox(label="处理状态", value="等待上传文件...", interactive=False, lines=2)
        with gr.Column(scale=4):
            with gr.Row():
                chatbot = gr.Chatbot(type="messages", height=500)
            with gr.Row():
                message = gr.Textbox(label="输入问题", placeholder="请输入问题", lines=1)
            with gr.Row():
                clear_history = gr.Button("🧹 清除历史对话")
                submit = gr.Button("🚀 提交")
        # 上传文件动作
        file.upload(fn=upload_file, inputs=file, outputs=status)
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