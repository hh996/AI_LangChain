from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    UnstructuredPDFLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)
from typing import List, Dict, Any, Optional
from tqdm import tqdm


class FileLoader:
    SUPPORTED_TYPES = {
        ".docx": Docx2txtLoader,
        ".pdf": UnstructuredPDFLoader,
        ".md": UnstructuredMarkdownLoader,
        ".txt": TextLoader,
    }

    def __init__(
        self,
        directory: str,
        file_types: Optional[List[str]] = None,
        use_multithreading: bool = True,
        silent_errors: bool = True,
        autodetect_encoding: bool = True,
        show_progress: bool = False,
    ):
        """
        初始化文件加载器

        :param directory: 文件目录路径
        :param file_types: 要加载的文件类型列表（如 [".docx", ".pdf"]），默认支持所有类型
        :param use_multithreading: 是否启用多线程加载
        :param silent_errors: 是否忽略加载错误
        :param autodetect_encoding: 是否自动检测编码（仅适用于文本文件）
        """
        self.directory = directory
        self.file_types = list(self.SUPPORTED_TYPES.keys())
        self.use_multithreading = use_multithreading
        self.silent_errors = silent_errors
        self.autodetect_encoding = autodetect_encoding

    def _get_loader_kwargs(self, ext: str) -> Dict[str, Any]:
        """根据文件扩展名生成对应的加载器参数"""
        if ext == ".txt" or ext == ".md":
            return {
                "autodetect_encoding": self.autodetect_encoding,
                "silent_errors": self.silent_errors,
            }
        return {}

    def _get_loader(self, ext: str):
        """根据文件扩展名获取对应的加载器类"""
        loader_cls = self.SUPPORTED_TYPES.get(ext)
        if not loader_cls:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader_cls

    def _build_loader(self, ext: str):
        """构建单个文件类型的加载器"""
        loader_cls = self._get_loader(ext)
        loader_kwargs = self._get_loader_kwargs(ext)
        return DirectoryLoader(
            self.directory,
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
            use_multithreading=self.use_multithreading,
        )

    def load(self) -> List[Any]:
        """加载所有指定类型的文件"""
        all_docs = []

        for ext in tqdm(self.file_types, desc="加载文件类型"):
            try:
                loader = self._build_loader(ext)
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                if not self.silent_errors:
                    raise
                print(f"跳过文件类型 {ext}: {e}")

        print(f"共加载 {len(all_docs)} 个文档。")
        return all_docs