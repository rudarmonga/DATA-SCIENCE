import os
import requests
import streamlit as st

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

from tavily import TavilyClient

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(
    page_title="City Agent",
    page_icon="🌍",
    layout="centered",
)

st.title("🌍 City Agent")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Settings")

approve_tools = st.sidebar.checkbox(
    "Approve all tool calls",
    value=True,
)

# ----------------------------
# Weather Tool
# ----------------------------
@tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN&appid={api_key}&units=metric"
    )

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


# ----------------------------
# Tavily
# ----------------------------
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str) -> str:
    """Get latest news about a city."""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3,
    )

    results = response.get("results", [])

    if not results:
        return "No news found."

    answer = ""

    for r in results:
        answer += (
            f"### {r.get('title')}\n"
            f"{r.get('content')}\n"
            f"{r.get('url')}\n\n"
        )

    return answer


# ----------------------------
# LLM
# ----------------------------
llm = ChatMistralAI(
    model="mistral-small-2506"
)


# ----------------------------
# Middleware
# ----------------------------
@wrap_tool_call
def human_approval(request, handler):

    if not approve_tools:
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"],
        )

    return handler(request)


# ----------------------------
# Agent
# ----------------------------
agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    middleware=[human_approval],
    system_prompt="""
You are a helpful city assistant.

Use get_weather for weather questions.

Use get_news for news questions.

Be concise.
""",
)

# ----------------------------
# Session State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Display Chat
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# Chat Input
# ----------------------------
prompt = st.chat_input("Ask me about any city...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = agent.invoke(
                {
                    "messages": st.session_state.messages
                }
            )

            answer = result["messages"][-1].content

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )