from ..graph import chatbot, pool
from langchain.messages import HumanMessage, ToolMessage, AIMessage


async def get_chat_history(thread_id):
    """Fetches history using the async graph state."""
    if chatbot is None:
        return []
    
    config = {'configurable': {'thread_id': str(thread_id)}}
    state = await chatbot.aget_state(config=config)
    
    if not (state and state.values):
        return []
    
    messages = state.values.get('messages', [])
    ui_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            ui_messages.append({'role': 'user', 'content': msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            ui_messages.append({'role': 'assistant', 'content': msg.content})
        elif isinstance(msg, ToolMessage):
            ui_messages.append({'role': 'tool', 'content': f"Executed tool: {str(msg.name)}"})
    return ui_messages


async def get_all_threads():
    """Retrieves all unique thread IDs using the async pool."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
            rows = await cur.fetchall()
            return [row[0] for row in rows]


async def delete_thread_history(thread_id):
    """Deletes all checkpoints for a thread using the async pool."""
    thread_id = str(thread_id)  
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
            await cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
            await cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
        await conn.commit()