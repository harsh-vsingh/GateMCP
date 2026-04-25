import os
import uuid
import json
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

#  Configuration & State 
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())

#  API Helpers 

def get_all_threads():
    try:
        return requests.get(f"{BACKEND_URL}/threads").json().get("threads", [])
    except Exception as e:
        st.error(f"Failed to fetch threads: {e}")
        return []

def get_chat_history(thread_id):
    try:
        return requests.get(f"{BACKEND_URL}/history/{thread_id}").json().get("history", [])
    except Exception as e:
        st.error(f"Failed to fetch chat history: {e}")
        return []

def delete_thread_history(thread_id):
    try:
        requests.delete(f"{BACKEND_URL}/history/{thread_id}")
    except Exception as e:
        st.error(f"Failed to delete thread: {e}")

import requests

def stream_chatbot_response(message, thread_id):
    try:
        with requests.post(
            f"{BACKEND_URL}/chat/stream",
            json={"message": message, "thread_id": thread_id},
            stream=True,
            timeout=120
        ) as r:

            if r.status_code != 200:
                st.error(f"HTTP Error {r.status_code}: {r.text}")
                return

            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        st.error("Invalid response from server")
                        break

                    if data["type"] == "error":
                        st.error(f"Backend Error: {data['content']}")
                        break

                    yield data["type"], data["content"]

    except requests.exceptions.Timeout:
        st.error("Request timed out")
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to backend")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")




#  UI Logic & Handlers 

def validate_mcp_json(config_json: str):
    try:
        config = json.loads(config_json)
        transport = config.get("transport")

        # Stdio
        if transport == "stdio":
            if not config.get("command"):
                return False, "Stdio requires a 'command'."
            if not isinstance(config["command"], str) or not config["command"].strip():
                return False, "Invalid 'command'."
            if "args" in config:
                if not isinstance(config["args"], list) or not all(isinstance(a, str) for a in config["args"]):
                    return False, "'args' must be a list of strings."
            if "url" in config:
                return False, "Stdio cannot have 'url'"

        # Remote
        elif transport in ["sse", "http", "streamable_http", "streamable-http"]:
            if not config.get("url"):
                return False, f"{transport} requires a 'url'."
            if not isinstance(config["url"], str) or not config["url"].strip():
                return False, "Invalid URL"
            if not config["url"].startswith(("http://", "https://")):
                return False, "URL must be http/https."

        # Unsupported
        else:
            return False, f"Unsupported transport: {transport}. Use 'stdio', 'sse', or 'http'."

        return True, config

    except json.JSONDecodeError as e:
        return False, f"Syntax Error: {str(e)}"
    except Exception as e:
        return False, f"Validation Error: {str(e)}"

def process_assistant_stream(stream_gen):
    with st.chat_message('assistant'):
        status_placeholder = st.container()
        response_placeholder = st.empty()
        full_response, active_tool_status = "", None

        for msg_type, content in stream_gen:
            if msg_type == "interrupt":
                if full_response: st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})
                st.session_state['interrupt_pending'] = content
                st.rerun()
            elif msg_type == "tool_start":
                with status_placeholder: active_tool_status = st.status(f"Running: {content}...", state="running", expanded=False)
                st.session_state['message_history'].append({'role': 'tool', 'content': f"Executed tool: {content}"})
            elif msg_type == "tool_end":
                if active_tool_status: active_tool_status.update(label=f"Finished: {content}", state="complete")
            elif msg_type == "content":
                full_response += content
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        if not full_response:
            st.warning("No response received from backend")
        if full_response: st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})

def trigger_approval(approved: bool):
    st.session_state['resume_approved'] = approved
    st.session_state['interrupt_pending'] = None
    st.rerun()

def reset_chat():
    st.session_state['thread_id'] = str(uuid.uuid4())
    st.session_state['message_history'] = []
    st.rerun()

#  Interactive Tool Management 

@st.dialog("GateMCP: Tool Management")
def manage_mcp_servers():
    st.markdown("### 🛠️ Active MCP Servers")
    try:
        servers = requests.get(f"{BACKEND_URL}/mcp/list").json()
        for name, config in servers.items():
            with st.container(border=True):
                c1, c2 = st.columns([0.7, 0.3])
                c1.markdown(f"**{name}**")
                c1.caption(f"Type: {config.get('transport')}")
                if name == "tls":
                    c2.button("System", disabled=True, key=f"lock_{name}", use_container_width=True)
                else:
                    if c2.button("🗑️ Delete", key=f"del_{name}", type="secondary", use_container_width=True):
                        requests.delete(f"{BACKEND_URL}/mcp/{name}")
                        st.rerun()
    except Exception: st.error("Failed to load servers.")

    st.divider()
    st.markdown("### ➕ Add New Server")
    new_name = st.text_input("Server Name")
    new_json = st.text_area("JSON Config", placeholder='{"transport": "http", "url": "..."}', height=100)
    
    if st.button("🚀 Connect & Sync", type="primary", use_container_width=True):
        is_valid, result = validate_mcp_json(new_json)
        if is_valid:
            resp = requests.post(f"{BACKEND_URL}/mcp/save", params={"name": new_name}, json=result)
            if resp.status_code == 200: st.success(f"'{new_name}' Live!"); st.rerun()
            else: st.error(resp.json().get("detail"))
        else: st.error(result)

#  Sidebar 

st.sidebar.title('🛡️ GateMCP Control')
if st.sidebar.button('➕ New Chat', use_container_width=True, type="primary"): reset_chat()
st.sidebar.divider()

if st.sidebar.button("🛠️ Manage Agent Tools", use_container_width=True): manage_mcp_servers()

with st.sidebar.expander("📚 Knowledge Base", expanded=False):
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Index Document", use_container_width=True):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        requests.post(f"{BACKEND_URL}/upload", files=files)
        st.success("Indexed!")

st.sidebar.divider()
st.sidebar.subheader('💬 Conversations')
st.session_state['chat_threads'] = get_all_threads()
for tid in st.session_state['chat_threads'][::-1]:
    c1, c2 = st.sidebar.columns([0.7, 0.3])
    if c1.button(str(tid)[:10], key=f"sel_{tid}", use_container_width=True):
        st.session_state['thread_id'], st.session_state['message_history'] = tid, get_chat_history(tid)
        st.rerun()
    if c2.button("❌", key=f"del_th_{tid}"):
        delete_thread_history(tid)
        if st.session_state['thread_id'] == tid: reset_chat()
        st.rerun()

#  Main Chat UI 

for msg in st.session_state['message_history']:
    if msg['role'] == 'tool': st.status(msg['content'], state="complete", expanded=False)
    else:
        with st.chat_message(msg['role']): st.markdown(msg['content'])

if user_input := st.chat_input('Message GateMCP...'):
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'): st.markdown(user_input)
    process_assistant_stream(stream_chatbot_response(user_input, st.session_state['thread_id']))

if st.session_state.get('resume_approved') is not None:
    appr = st.session_state.pop('resume_approved')
    def approval_gen():
        try:
            with requests.post(
                f"{BACKEND_URL}/chat/approve",
                json={"thread_id": st.session_state['thread_id'], "approved": appr},
                stream=True
            ) as r:

                if r.status_code != 200:
                    st.error(f"HTTP Error {r.status_code}: {r.text}")
                    return

                for line in r.iter_lines():
                    if line:
                        try:
                            d = json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            st.error("Invalid response from server")
                            break

                        if d["type"] == "error":
                            st.error(f"Backend Error: {d['content']}")
                            break

                        yield d["type"], d["content"]

        except Exception as e:
            st.error(f"Approval request failed: {e}")
    process_assistant_stream(approval_gen())

if st.session_state.get('interrupt_pending'):
    t_name = st.session_state['interrupt_pending']
    with st.chat_message("assistant"):
        st.warning(f"Permission Requested: Run **{t_name}**?")
        col1, col2 = st.columns(2)
        if col1.button("✅ Approve", key="approve_hitl", use_container_width=True): trigger_approval(True)
        if col2.button("❌ Deny", key="deny_hitl", use_container_width=True): trigger_approval(False)