import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from code_generator_1_3b import CodeGenerator
from chat_codegen_wrapper import ChatCodeGenerator

st.title("Unreal Code Generator")

if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.user_history = []
    st.rerun()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_history" not in st.session_state:
    st.session_state.user_history = []

# 이전 대화 출력
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


@st.cache_resource
def get_model():
    generator = CodeGenerator(
        load_path="/home/ubuntu/checkpoints/deepseek_lora",
        device="cuda",
        quantization="4bit",
        lora=True,
        max_length=1024,
        max_new_tokens=512,
    )
    return ChatCodeGenerator(generator)

model = get_model()

# 사용자 입력 처리
if prompt := st.chat_input("Enter your Unreal Engine C++ requirement."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.user_history.append(HumanMessage(content=prompt))

    # 최근 사용자 메시지 3개만 전달
    user_messages = st.session_state.user_history[-3:]

    # 모델 호출
    result = model.invoke(user_messages)

    with st.chat_message("assistant"):
        st.markdown(result.content)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.content
        })
