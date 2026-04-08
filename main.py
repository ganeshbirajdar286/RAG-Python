from  dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai  import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorStore=Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever=vectorStore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":4,
                   "fetch_k":10, # first it search with similarity and find 10 records and then applys the mmr  of 10 records  and find best  4
                    "lambda_mult":0.5 # tell about diversed result .it's value are from 0 to 1 and if 0 no diversed ans  
                   }
)

llm=ChatMistralAI(model="mistral-small-2506")

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("RAG system created ")

print("press 0 to exit")

while True:
    query=input("you:")
    if query =="0":
        break;
    else :
        docs=retriever.invoke(query)
        context="\n\n".join(
            [doc.page_content for doc  in docs]
        )

    final_prompt=prompt.invoke({
        "context":context,"question":query
    })

    response =llm.invoke(final_prompt)
    print(f"]\n AI:{response.content} ")