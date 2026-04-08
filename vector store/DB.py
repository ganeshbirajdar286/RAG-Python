from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document # create documents 

load_dotenv()
# Step 1: Create documents
docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]# use for create document not as use 

#chroma db create embedding onit own .only thing what we  need to do is telling which model to  use

# Step 2: Load embedding model
embeddings_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
# Step 3: Create Chroma DB
db=Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    persist_directory="./chroma_db"
)


result=db.similarity_search("what is used for data  analysis?",k=2) #k tell us about how much document  to  return
 
for r in result:
    print(r.page_content)
    print(r.metadata)

retriver=db.as_retriever()
docs =retriver.invoke("Explain deep learning ")

for d in docs:
    print(d.page_content)