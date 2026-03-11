import streamlit as st



def create_memory():
    """Initialize chat memory in Streamlit session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history



def load_memory_text():
    """Return conversation history as formatted text."""
    history = st.session_state.get("chat_history", [])

    memory_text = ""
    for turn in history:
        memory_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    return memory_text



def save_turn(user_message, assistant_message):
    """Save a chat turn to memory."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({
        "user": user_message,
        "assistant": assistant_message
    })
