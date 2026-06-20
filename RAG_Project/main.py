from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = MistralAIEmbeddings(model = "mistral-embed")

vectorStore = Chroma(
    persist_directory ="chroma-db",
    embedding_function = embedding_model
)

retriever = vectorStore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k" : 10,
        "lambda_mult" : 0.5
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")

template = ChatPromptTemplate.from_messages([
    ("system", 
     """
     You are a helpfull AI Assistant.
     Use ONLY the provided context to answer the question.
     If the answer is not present in the context,
     Say: "I could not find answer in the document."
     """
    ),
    ("human", 
     """
     context: {context},
     Query: {question}
     """)
])

print("Rag System Created")

print("press 0 to exit ")

while True :
    query = input("You: ")
    if query == "0" : 
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = template.invoke({
        "context" : context,
        "question" : query
    })
    
    result = llm.invoke(final_prompt)
    print("\n\n",result.content)