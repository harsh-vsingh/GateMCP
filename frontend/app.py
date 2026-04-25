import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import requests
import uuid
import json


BACKEND_URL = os.getenv("BACKEND_URL")

def get_all_threads():
    try:
        response = requests.get(f"{BACKEND_URL}/threads")
        return response.json().get("threads", [])
    except Exception:
        return []

def get_chat_history(thread_id):
    try:
        response = requests.get(f"{BACKEND_URL}/history/{thread_id}")
        return response.json().get("history", [])
    except Exception:
        return []

def delete_thread_history(thread_id):
    try:
        requests.delete(f"{BACKEND_URL}/history/{thread_id}")
    except Exception:
        pass

def stream_chatbot_response(message, thread_id):
    """Call the backend streaming endpoint."""
    try:
        with requests.post(
            f"{BACKEND_URL}/chat/stream", 
            json={"message": message, "thread_id": thread_id}, 
            stream=True,
            timeout=120 
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if data["type"] == "error":
                        st.error(f"Backend Error: {data['content']}")
                        break
                    yield data["type"], data["content"]
    except Exception as e:
        st.error(f"Connection Error: {e}")



def process_assistant_stream(stream_gen):
    """Unified handler for streaming assistant responses and tools."""
    with st.chat_message('assistant'):
        status_placeholder = st.container()
        response_placeholder = st.empty()
        full_response = ""
        active_tool_status = None

        for msg_type, content in stream_gen:
            if msg_type == "interrupt":
                # Save partial progress before rerunning
                if full_response:
                    st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})
                st.session_state['interrupt_pending'] = content
                st.rerun()

            elif msg_type == "tool_start":
                with status_placeholder:
                    active_tool_status = st.status(f"Running: {content}...", state="running", expanded=False)
                st.session_state['message_history'].append({'role': 'tool', 'content': f"Executed tool: {content}"})
            
            elif msg_type == "tool_end":
                if active_tool_status:
                    active_tool_status.update(label=f"Finished: {content}", state="complete", expanded=False)

            elif msg_type == "content":
                full_response += content
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
        if full_response:
            st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})


def trigger_approval(approved: bool):
    """Sets the state to resume and clears the interrupt."""
    st.session_state['resume_approved'] = approved
    st.session_state['interrupt_pending'] = None
    # We don't call process_assistant_stream here to avoid the white screen
    st.rerun()



@st.dialog("MCP Server Manager")
def show_mcp_dialog():
    st.write("### Active Configurations")
    try:
        servers = requests.get(f"{BACKEND_URL}/mcp/list").json()
        for name, config in servers.items():
            c1, c2 = st.columns([0.8, 0.2])
            c1.code(f"{name}: {config.get('transport')}", language="text")
            if c2.button("🗑️", key=f"del_{name}"):
                requests.delete(f"{BACKEND_URL}/mcp/{name}")
                st.rerun()
    except Exception:
        st.error("Could not fetch server list.")


# frontend/app.py additions

def mcp_tool_manager():
    st.sidebar.divider()
    if st.sidebar.button("🛠️ Manage MCP Servers"):
        show_mcp_dialog()

@st.dialog("MCP Server Manager")
def show_mcp_dialog():
    st.write("### Active Configurations")
    # Fetch list from backend
    servers = requests.get(f"{BACKEND_URL}/mcp/list").json()
    
    for name, config in servers.items():
        c1, c2 = st.columns([0.8, 0.2])
        c1.code(f"{name}: {config.get('transport')}", language="text")
        if c2.button("🗑️", key=f"del_{name}"):
            requests.delete(f"{BACKEND_URL}/mcp/{name}")
            st.rerun()

    st.write("---")
    st.write("### Add New Server")
    new_name = st.text_input("Server Name (e.g., 'weather-api')")
    new_json = st.text_area("JSON Configuration", placeholder='{"transport": "sse", "url": "..."}')
    
    if st.button("Save Server"):
        try:
            config = json.loads(new_json)
            requests.post(f"{BACKEND_URL}/mcp/save", params={"name": new_name}, json=config)
            st.success(f"Saved {new_name} to workspace.")
            st.rerun()
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    st.divider()
    if st.button("🔄 Sync & Rebuild Agent", type="primary", use_container_width=True):
        with st.spinner("Kicking off hot-reload..."):
            resp = requests.post(f"{BACKEND_URL}/mcp/refresh")
            if resp.status_code == 200:
                st.success("Graph Re-compiled with new tools!")
                st.rerun()

# Helpers
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    st.rerun()

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
         st.session_state['chat_threads'].append(thread_id)



# Initialize
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

if 'thread_id' not in st.session_state:
    if st.session_state['chat_threads']:
        st.session_state['thread_id'] = st.session_state['chat_threads'][0]
        st.session_state['message_history'] = get_chat_history(st.session_state['thread_id'])
    else:
        st.session_state['thread_id'] = str(uuid.uuid4())

add_thread(st.session_state['thread_id'])



# UI setup
st.sidebar.title('LangGraph Chatbot')

uploaded_file = st.sidebar.file_uploader("Upload PDF to Vector DB", type="pdf")

if uploaded_file:
    if st.sidebar.button("Process File"):
        with st.sidebar.spinner("Ingesting PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{BACKEND_URL}/upload", files=files)
            
            if response.status_code == 200:
                st.sidebar.success(f"Successfully indexed {uploaded_file.name}!")
            else:
                st.sidebar.error("Failed to index file.")

if st.sidebar.button("🔄 Update Tools"):
    with st.sidebar.spinner("Rebuilding Graph..."):
        resp = requests.post(f"{BACKEND_URL}/mcp/refresh")
        st.sidebar.success("Agent Tools Updated!")

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    cols = st.sidebar.columns([0.8, 0.2])
    
    if cols[0].button(str(thread_id)[:20], key=f"select_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        st.session_state['message_history'] = get_chat_history(thread_id)
        st.rerun()

    if cols[1].button("❌", key=f"delete_{thread_id}"):
        delete_thread_history(thread_id)
        st.session_state['chat_threads'].remove(thread_id)
        if st.session_state.get('thread_id') == thread_id:
            reset_chat()
        st.rerun()




# Chat interface
for message in st.session_state['message_history']:
    if message['role'] == 'tool':
        st.status(message['content'], state="complete", expanded=False)
    else:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    process_assistant_stream(stream_chatbot_response(user_input, st.session_state['thread_id']))

if st.session_state.get('resume_approved') is not None:
    approved = st.session_state.pop('resume_approved')
    
    def approval_gen():
        with requests.post(
            f"{BACKEND_URL}/chat/approve",
            json={"thread_id": st.session_state['thread_id'], "approved": approved},
            stream=True
        ) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    yield data["type"], data["content"]

    process_assistant_stream(approval_gen())

if st.session_state.get('interrupt_pending'):
    tool_name = st.session_state['interrupt_pending']
    hitl_placeholder = st.empty()
    
    with hitl_placeholder.container():
        with st.chat_message("assistant"):
            st.warning(f"Permission Requested: Run **{tool_name}**?")
            col1, col2 = st.columns(2)
            
            if col1.button("✅ Approve", key="approve_hitl"):
                trigger_approval(True) 
                
            if col2.button("❌ Deny", key="deny_hitl"):
                trigger_approval(False)