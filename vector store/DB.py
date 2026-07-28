from json import load

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs=[
Document(page_content="Python is a widely used in artificial intelligence.", metadata={"source":"test document"}),
Document(page_content="Pandas is used for data analysis in Python.", metadata={"source":"another test document"}),    
Document(page_content="Neural networks are used in deep learning.", metadata={"source":"third test document"})
]


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma-db"
)

result=vectorstore.similarity_search("What is used for data analysis?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)


retriever=vectorstore.as_retriever()

docs=retriever.invoke("Explain deep learning.")

for d in docs:
    print(d.page_content)
