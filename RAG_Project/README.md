# 🧠 Deep Learning Knowledge Assistant

A simple RAG (Retrieval-Augmented Generation) application built using LangChain, Mistral AI, ChromaDB, and Streamlit.

This project allows users to ask questions about Deep Learning and Machine Intelligence using information stored in a vector database.

## 📚 Documents Used

* Fundamentals of Deep Learning
* Designing Next-Generation Machine Intelligence Algorithms

## 🚀 Features

* Ask questions about the documents
* Retrieve relevant information using ChromaDB
* Generate answers using Mistral AI
* Simple Streamlit chat interface
* Display retrieved document chunks

## 🛠️ Technologies Used

* Python
* LangChain
* Mistral AI
* ChromaDB
* Streamlit

## 📂 Project Structure

```text
RAG_Project/
│
├── app.py
├── main.py
├── Create_DB.py
├── requirements.txt
├── .env
├── chroma-db/
├── document_loader/
└── vectorStore/
```

## ⚙️ Setup

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Add API Key

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key_here
```

### Create Vector Database

```bash
python Create_DB.py
```

### Run Application

```bash
streamlit run app.py
```

## 💡 Example Questions

* What is Deep Learning?
* What is Transfer Learning?
* Explain Neural Networks.
* What is Gradient Descent?

## 🔮 Future Improvements

* Add PDF upload support
* Improve UI design
* Add source citations
* Add chat history saving
* Support more documents
* Add document statistics in sidebar

## 📖 Learning Outcomes

Through this project, I learned:

* LangChain basics
* Vector databases with ChromaDB
* Embeddings
* Retrieval-Augmented Generation (RAG)
* Streamlit application development
* Working with Mistral AI APIs

```
```
