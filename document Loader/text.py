from langchain_community.document_loaders  import TextLoader
from langchain_text_splitters import CharacterTextSplitter


data=TextLoader("notes.txt")
text_splitter = CharacterTextSplitter(
     separator="", #default separator ="\n\n"
    chunk_size=10, 
    chunk_overlap=1
    )
docs=data.load()
chunks= text_splitter.split_documents(docs)
print(len(chunks))

for i in chunks:
    print(i.page_content)