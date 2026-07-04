from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_result = 3)

model = ChatMistralAI(model = "mistral-small-2506")

prompt = ChatPromptTemplate.from_template(
    """
    You are a helful AI assistant.
    Summarize the following news into clear bullet points.
    {news}
    """
)

chain = prompt | model | StrOutputParser()

new_result = search_tool.invoke("Latest AI news in 2026")

result = chain.invoke({"news" : new_result})

print(result)