from dotenv import  load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders  import PyPDFLoader
from langchain_core.prompts  import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

model = ChatMistralAI(model = "mistral-small-2506",temperature=0);

template =ChatPromptTemplate.from_messages([
    ("system","you are the ai for summarly text" ),
    ("human","{data}")
])

data=PyPDFLoader("deep.pdf")
docs =data.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chucks=text_splitter.split_documents(docs)

print(chucks[0].page_content)

prompt =template.format_messages(data = chucks[0].page_content)

result =model.invoke(prompt)
print(result.content)
