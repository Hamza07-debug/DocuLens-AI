from json import load

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from openai import chat

load_dotenv()

embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model 
)

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5  
    }
)

llm=ChatMistralAI(model_name="mistral-small-2603", temperature=0.1)


# prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",
        """You are a helpful assistant
         that answers questions based on the context provided.
         If you don't know the answer, just say you don't know.
           Do not try to make up an answer."""),
        ("human","""Context:
        {context}

        Question:
        {question}""")
        
    ]
)

print("Welcome to the RAG system. Type 'exit' to quit.")

while True:
    query=input("Enter your question: ")
    if query.lower()=="exit":
        break

    docs=retriever.invoke(query)

    context="\n".join([d.page_content for d in docs])

    response=llm.invoke(
        prompt.format_prompt(
            context=context,
            question=query
        ).to_messages()
    )

    print("Answer:", response.content)