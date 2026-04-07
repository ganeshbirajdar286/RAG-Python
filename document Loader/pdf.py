from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter


loader = PyPDFLoader("GRU.pdf")
docs = loader.load()
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
chucks=text_splitter.split_documents(docs)

print(len(chucks))

print(chucks[0].page_content)


