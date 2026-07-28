from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data=PyPDFLoader("document loaders/AI_Info.pdf")

docs=data.load()

splitter=RecursiveCharacterTextSplitter(
chunk_size=4000,
  chunk_overlap=10
)

chunks=splitter.split_documents(docs)

print(len(chunks))

print(chunks[0].page_content)





#print(docs[0])




