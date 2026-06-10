from langchain_community.document_loaders import PyMuPDFLoader

pdf_path = "D:\Workspace\PycharmProjects\PythonProject\k-ai-knowledge\smar_reading\data\sample_document .pdf"

pdf_loader = PyMuPDFLoader(pdf_path)
docs = pdf_loader.load() # 将pdf文档加载到内存里面来！docs有文档内容
print(type(docs)) # <class 'list'>
print(len(docs)) # 6
print(type(docs[0])) # <class 'langchain_core.documents.base.Document'>

"""
page_content: 文档内容
metadata：元数据 page
"""
for doc in docs:
    print(doc)