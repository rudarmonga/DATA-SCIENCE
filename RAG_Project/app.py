
from dotenv import load_dotenv
import streamlit as st

from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="Deep Learning Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_rag_system():

    embedding_model = MistralAIEmbeddings(
        model="mistral-embed"
    )

    vector_store = Chroma(
        persist_directory="chroma-db",
        embedding_function=embedding_model
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful AI Assistant.

                Answer ONLY using the provided context.

                If the answer is not present in the context,
                respond exactly with:

                "I could not find answer in the document."
                """
            ),
            (
                "human",
                """
                Context:
                {context}

                Question:
                {question}
                """
            )
        ]
    )

    return retriever, llm, prompt


retriever, llm, prompt = load_rag_system()

with st.sidebar:

    st.title("📚 Knowledge Base")

    st.success("Documents Loaded")

    st.markdown("""
    ### Available Documents

    ✅ Fundamentals of Deep Learning

    ✅ Designing Next-Generation Machine Intelligence Algorithms
    """)

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🧠 Deep Learning Knowledge Assistant")

st.caption(
    "Ask questions about Deep Learning, Neural Networks, AI Systems, and Machine Intelligence."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about your documents...")

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            docs = retriever.invoke(query)

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            result = llm.invoke(final_prompt)

            answer = result.content

            st.markdown(answer)

            with st.expander("📖 Retrieved Context"):

                for idx, doc in enumerate(docs, start=1):

                    st.markdown(f"### Chunk {idx}")

                    st.write(doc.page_content)

                    st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
