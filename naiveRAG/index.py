from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader, \
    UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import re


# 加载数据集
def load_data() -> list[Document]:
    # 加载文本数据
    text_documents: list[Document] = TextLoader(file_path="./knowledge_base/sample.txt", encoding="utf-8").load()
    # 加载 md
    md_documents: list[Document] = UnstructuredMarkdownLoader(file_path="./knowledge_base/sample.md", mode="single",
                                                              strategy="hi_res").load()
    # 加载 docx
    docx_documents: list[Document] = UnstructuredWordDocumentLoader(file_path="./knowledge_base/sample.docx",
                                                                    mode="single", strategy="hi_res").load()
    # 加载 PDF
    pdf_documents: list[Document] = UnstructuredPDFLoader(file_path="./knowledge_base/sample.pdf", mode="elements",
                                                          strategy="hi_res", infer_table_structure=True,
                                                          languages=["eng", "chi_sim"]).load()
    return text_documents + md_documents + docx_documents + pdf_documents


# 清洗数据
def clean_documents(documents: list[Document]) -> list[Document]:
    result: list[Document] = []
    for document in documents:
        # 清洗内容
        content = document.page_content
        content = content.strip()
        content = re.sub(r"\n{2,}", "\n", content)
        document.page_content = content

        # chroma 支持的类型
        chroma_db_support_type = (int, float, str, bool)
        for key, value in document.metadata.items():
            if not isinstance(value, chroma_db_support_type):
                try:
                    document.metadata[key] = json.dumps(value, ensure_ascii=False)
                except:
                    print(key, "序列化失败")
                    document.metadata[key] = str(value)

        result.append(document)
    return result


# 切片
def split_documents(documents: list[Document]) -> list[Document]:
    result: list[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=500, chunk_overlap=200)
    result = text_splitter.split_documents(documents)
    return result


# 构建索引
# 加载到 ChromaDB 中
def getEmbdding() -> HuggingFaceEmbeddings:
    model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )
    return model


def save_chromadb(documents: list[Document], model: Embeddings):
    Chroma.from_documents(documents=documents, embedding=model, persist_directory="./vectorstore")

if  __name__ == "__main__":
    document: list[Document] = load_data()
    clean_data: list[Document] = clean_documents(document)
    save_chromadb(clean_data, getEmbdding())