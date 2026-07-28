from langchain_community.document_loaders import WebBaseLoader

url="https://www.apple.com/"

data=WebBaseLoader(url)

docs=data.load()

print(len(docs))

print(docs[0].page_content[:1000])