import json
from .. import graph
from langchain.messages import ToolMessage

def serialize_content(content):
    """Converts complex message content into a clean string for the UI."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in content])
    return str(content)

async def get_streaming_response(message: str, thread_id: str, is_resume: bool = False):
    config = {"configurable": {"thread_id": thread_id}}
    input_data = None if is_resume else {"messages": [("user", message)]}

    # Retry full stream on failure
    MAX_RETRIES = 2
    if graph.chatbot is None:
        yield json.dumps({
            "type": "system_error",
            "content": "Chatbot not initialized"
        }) + "\n"
        return
    for attempt in range(MAX_RETRIES):
        last_event_type = None
        current_tool = None
        try:
            # Streaming chatbot response
            async for event in graph.chatbot.astream_events(input_data, config, version="v2"):
                kind = event["event"]
                last_event_type = kind

                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield json.dumps({
                            "type": "content",
                            "content": serialize_content(content)
                        }) + "\n"

                elif kind == "on_tool_start":
                    current_tool = event["name"]
                    yield json.dumps({
                        "type": "tool_start",
                        "content": event["name"]
                    }) + "\n"

                elif kind == "on_tool_end":
                    yield json.dumps({
                        "type": "tool_end",
                        "content": event["name"]
                    }) + "\n"

            # Check if if stream pause for tool approval
            state = await graph.chatbot.aget_state(config)
            if state.next and state.next[0] == "tools":
                last_msg = state.values["messages"][-1]
                tool_name = last_msg.tool_calls[0]["name"] if last_msg.tool_calls else "tool"
                yield json.dumps({
                    "type": "interrupt",
                    "content": tool_name
                }) + "\n"
            return 

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                continue

            # classify error based on last event
            err_str = str(e).lower()
            if last_event_type and last_event_type.startswith("on_tool"):
                err_type = "tool_error"
            elif last_event_type and last_event_type.startswith("on_chat_model"):
                err_type = "llm_error"
            elif "psycopg" in err_str or "connection" in err_str or "cursor" in err_str:
                err_type = "db_error"
            else:
                err_type = "system_error"
            playload = {
                "type": err_type,
                "content": str(e)
            }
            if err_type == "tool_error" and current_tool:
                playload["tool"] = current_tool

            yield json.dumps(playload) + "\n"

async def handle_denial(thread_id: str):
    """Injects a cancellation message so the LLM knows the user denied the tool."""
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.chatbot.aget_state(config)
    
    tool_call_id = state.values["messages"][-1].tool_calls[0]["id"]
    cancellation_msg = ToolMessage(
        tool_call_id=tool_call_id,
        content="User denied permission to execute this tool."
    )
    await graph.chatbot.aupdate_state(config, {"messages": [cancellation_msg]}, as_node="tools")