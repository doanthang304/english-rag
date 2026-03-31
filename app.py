from __future__ import annotations

from uuid import uuid4

import streamlit as st

from engine import EducationAgenticRAG

st.set_page_config(
    page_title="Agentic RAG giáo dục",
    layout="centered",
)

# 2. CSS TỐI GIẢN
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 600;
            color: #1F2937; /* Xám đậm thanh lịch */
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #6B7280; /* Xám nhạt */
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "agent" not in st.session_state:
    st.session_state.agent = EducationAgenticRAG()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent: EducationAgenticRAG = st.session_state.agent

with st.sidebar:
    st.title(" Bảng Điều Khiển")
    
    # Nhóm Cấu hình
    with st.expander(" Cấu hình hệ thống", expanded=True):
        model_name = st.text_input("Mô hình Groq", value=agent.default_model)
        web_search_enabled = st.toggle("Tra cứu Web", value=True)
        thread_id = st.text_input("Thread ID", value=st.session_state.thread_id)
        if thread_id != st.session_state.thread_id:
            st.session_state.thread_id = thread_id

    with st.expander("Nguồn dữ liệu", expanded=False):
        st.markdown(
            """
            **Cấu trúc thư mục:**
            - Đặt SGK/đề cũ (`.pdf`, `.md`, `.txt`)
            - Đặt ngân hàng câu hỏi (`.csv`, `.json`)
            """
        )
        if st.button("Làm mới dữ liệu", use_container_width=True):
            with st.spinner("Đang nạp PDF, ngân hàng câu hỏi, vector/metadata store..."):
                total_chunks = agent.ingest_documents()
            st.success(f"Đã đồng bộ {total_chunks} chunks.")
            
    with st.expander(" Gợi ý câu lệnh", expanded=False):
        st.markdown(
            """
            - Soạn giáo án Tiếng Anh lớp 4, bộ KNTT, Unit 5, cơ bản.
            - Tạo đề kiểm tra giữa kỳ Tiếng Anh lớp 8, bộ CTST, Unit 3, trung bình.
            - Soạn worksheet Reading cho Tiếng Anh lớp 11, bộ CĐ, Unit 2, nâng cao.
            """
        )

# 5. KHU VỰC CHÍNH - HEADER & TRÒ CHUYỆN
st.markdown('<div class="main-header">Trợ lý Giáo dục Thông minh</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Soạn giáo án, đề thi và thiết kế học liệu chuẩn xác</div>', unsafe_allow_html=True)

# Hiển thị lịch sử trò chuyện
for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

# Ô nhập liệu chat
user_input = st.chat_input("Nhập yêu cầu (VD: Soạn đề thi lớp 10, bộ KNTT, Unit 1, mức độ khó...)")

if user_input:
    # Thêm câu hỏi vào lịch sử và hiển thị
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Xử lý phản hồi từ Agent
    with st.chat_message("assistant"):
        with st.spinner("Đang lập kế hoạch, truy xuất dữ liệu và sinh nội dung..."):
            try:
                result = agent.invoke(
                    query=user_input,
                    thread_id=st.session_state.thread_id,
                    model_name=model_name,
                    web_search_enabled=web_search_enabled,
                )
                answer = result["final_answer"]
                
                # Hiển thị câu trả lời
                st.markdown(answer)
                
                # Giấu chi tiết kỹ thuật vào expander gọn gàng
                with st.expander(" Xem luồng xử lý của Agent"):
                    st.json(
                        {
                            "planner_output": result.get("planner_output", {}),
                            "local_context": result.get("local_context", ""),
                            "web_context": result.get("web_context", ""),
                            "review_output": result.get("review_output", {}),
                        }
                    )
                    
                # Lưu vào lịch sử
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
            except Exception as exc:
                error_message = f"**Lỗi xử lý:** {exc}"
                st.error(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})