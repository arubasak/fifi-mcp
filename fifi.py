import streamlit as st
import datetime
import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

# --- Load environment variables from secrets ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MCP_PINECONE_URL = st.secrets["MCP_PINECONE_URL"]
MCP_PINECONE_API_KEY = st.secrets["MCP_PINECONE_API_KEY"]
MCP_PIPEDREAM_URL = st.secrets["MCP_PIPEDREAM_URL"]

# --- LangChain LLM (OpenAI GPT-4o) ---
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# --- Async function to initialize LangChain Agent ---
async def initialize_langchain_agent():
    # MultiServerMCPClient setup
    client = MultiServerMCPClient(
        {
            "pinecone": {
                "url": MCP_PINECONE_URL,
                "transport": "sse",
                "headers": {"Authorization": f"Bearer {MCP_PINECONE_API_KEY}"}
            },
            "pipedream": {
                "url": MCP_PIPEDREAM_URL,
                "transport": "sse"
            }
        }
    )

    # Await the tools from MCP Client
    tools = await client.get_tools()

    # Create agent with the tools
    agent = create_react_agent(llm, tools)
    return agent

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Initialize agent (this is async) ---
if 'agent' not in st.session_state:
    st.session_state.agent = asyncio.run(initialize_langchain_agent())

# --- Async handler for user queries ---
async def handle_user_query_async(user_query: str):
    if not user_query:
        return
    st.session_state.messages.append({"role": "user", "content": user_query})

    try:
        with st.spinner("FiFi Co-Pilot is thinking..."):
            result = await st.session_state.agent.ainvoke({"messages": [{"role": "user", "content": user_query}]})
        assistant_reply = result["messages"][-1].content
    except Exception as e:
        assistant_reply = f"(Error: {e})"

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# --- Streamlit UI ---
st.title("FiFi Co-Pilot 🚀 (LangChain MCP Agent)")

st.sidebar.markdown("## Quick Questions")
preview_questions = [
    "Help me with my recipe for a new juice drink",
    "Suggest me some strawberry flavours for beverage",
    "I need vanilla flavours for ice-cream"
]
for question in preview_questions:
    if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
        asyncio.run(handle_user_query_async(question))

user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input")
if user_prompt:
    asyncio.run(handle_user_query_async(user_prompt))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(str(message.get("content", "")))

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []

if st.session_state.messages:
    chat_export_data_txt = "\n\n".join(
        [f"{str(msg.get('role', 'Unknown')).capitalize()}: {str(msg.get('content', ''))}" for msg in st.session_state.messages]
    )
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.sidebar.download_button(
        label="📥 Download Chat (TXT)",
        data=chat_export_data_txt,
        file_name=f"fifi_mcp_chat_{current_time}.txt",
        mime="text/plain",
        use_container_width=True
    )

st.sidebar.markdown("---")
st.sidebar.info("💡 FiFi will use dynamic tool calling to decide which MCP to talk to – no hints needed in the UI!")
