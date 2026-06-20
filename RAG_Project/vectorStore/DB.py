from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

data = PyPDFLoader("document_loader/deeplearning.pdf")
docs = data.load()

embedding_modle = MistralAIEmbeddings(model = "mistral-embed")

vectorStore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_modle,
    persist_directory = "chroma-db"
)

result = vectorStore.similarity_search("What is Recurrent Neural Network?")

for r in result:
    print(r.page_content)
    print(r.metadata)
    print()
    print()
    print()

print("Search Done")
retriver = vectorStore.as_retriever()

docs = retriver.invoke("Explain Deep Learning")

for d in docs:
    print(d.page_content)
    print(d.metadata)
    print()
    print()
    print()