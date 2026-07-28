from langchain_community.document_loaders import PyPDFLoader

data=PyPDFLoader("document loaders/AI_Info.pdf")

docs=data.load()

#print(docs)

print(len(docs))

print(docs[247])

