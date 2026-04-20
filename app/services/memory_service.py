
from langchain_classic.memory import ConversationBufferMemory

class MemoryManager:
    def __init__(self):
        # in-memory map: session_id -> ConversationBufferMemory
        self._map = {}

    def get_memory(self, session_id: str):
        if not session_id:
            return
        if session_id not in self._map:
            self._map[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")
        else:
        # auto-truncate
            mem = self._map[session_id]
            if len(mem.chat_memory.messages) > 20:    # giới hạn 40 message
                mem.chat_memory.messages = mem.chat_memory.messages[-10:]  # giữ lại 20 cuối
        return self._map[session_id]

    def remove_memory(self, session_id: str):
        if session_id in self._map:
            del self._map[session_id]