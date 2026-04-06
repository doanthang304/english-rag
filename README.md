*English-Agentic-RAG: Trợ lý Sư phạm Thông minh
- Dự án này là một hệ thống Agentic RAG tiên tiến, được thiết kế riêng để hỗ trợ giáo viên tiếng Anh (tập trung vào bộ sách Global Success). Khác với các hệ thống RAG truyền thống, dự án này sử dụng luồng công việc Stateful Multi-Agent để đảm bảo tính chính xác về mặt sư phạm và khả năng tự sửa lỗi (Self-correction).

***Tính năng nổi bật (Key Features)***
- Agentic Reasoning Loop: Sử dụng kiến trúc Planner-Critic. Agent không chỉ trả lời ngay lập tức mà thực hiện quy trình: Lập kế hoạch -> Truy xuất -> Tạo nội dung -> Kiểm định lại để đảm bảo đúng tiêu chuẩn kiến thức lớp 7. 
- Pedagogical Grounding: Sử dụng kỹ thuật Metadata Filtering để đảm bảo AI chỉ sử dụng từ vựng và ngữ pháp trong phạm vi các Unit/Grade đã được định nghĩa trong knowledge_base.
- Context-Aware Retrieval: Tích hợp BGE-M3 Multilingual Embeddings để đạt độ chính xác cao khi tìm kiếm ngữ nghĩa (Semantic Search) cho cả tiếng Anh và tiếng Việt.
- Stateful Conversations: Quản lý hội thoại thông qua LangGraph Checkpoints, cho phép AI ghi nhớ ngữ cảnh trong các phiên làm việc phức tạp (Multi-turn conversation).
- Real-time Web Augmentation: Kết hợp công cụ tìm kiếm DuckDuckGo Search để cập nhật các xu hướng giáo dục và mẫu giáo án mới nhất từ internet.

***Kiến trúc hệ thống (System Architecture)***
- Hệ thống hoạt động dựa trên mô hình Directed Acyclic Graph (DAG):
- Planner Node: Phân tích yêu cầu người dùng (Intent) thành một cấu trúc dữ liệu cụ thể (Grade, Unit, Difficulty, Task Type).
- Retriever Node: Thực hiện tìm kiếm hỗn hợp (Hybrid Search) trên ChromaDB kết hợp với bộ lọc Metadata.
- Generator Node: Soạn thảo nội dung (Giáo án, Đề thi, hoặc Worksheet).
- Critic/Validator Node: Đánh giá bản thảo dựa trên các ràng buộc về chương trình học. Nếu phát hiện lỗi (ví dụ: từ vựng quá khó), hệ thống sẽ kích hoạt vòng lặp (Loop) quay lại Generator để chỉnh sửa.

***Tech Stack***

- Framework chính: LangGraph & LangChain
- Inference Engine: Groq Cloud (Model: llama-3.1-8b-instant)
- Vector Database: ChromaDB
- Embedding Model: BAAI/bge-m3 (Chạy Local)

Giao diện (UI): Streamlit

Xử lý dữ liệu: PyPDF, RecursiveCharacterTextSplitter

***Hướng dẫn khởi chạy (Getting Started)***
1. Yêu cầu hệ thống
Python 3.10 trở lên.

Groq API Key (Lấy tại console.groq.com).

2. Cài đặt (Installation)
# Clone repository này
git clone https://github.com/doanthang304/english-rag
cd English-Agentic-RAG

# Tạo và kích hoạt môi trường ảo (Virtual Environment)
python -m venv venv
.\venv\Scripts\activate # Đối với Windows

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

3. Cấu hình (Configuration)
cp .env.example .env

4. Data Ingestion
Đặt các file tài liệu (.pdf, .md, .txt) vào thư mục data/knowledge_base/ và chạy lệnh:
python -m src.english_agent.ingestion

5. Chạy ứng dụng
python -m streamlit run app.py
 ----------------------------------------------------------------------------

***Cấu trúc dự án (Project Structure)

```bash 
EnglishAgent/
├── src/
│   └── english_agent/
│       ├── graph.py       # Định nghĩa luồng công việc LangGraph
│       ├── ingestion.py   # Logic xử lý dữ liệu và cấu hình Vector DB
│       ├── prompts.py     # Quản lý System Instructions và Personas
│       ├── schemas.py     # Định nghĩa Pydantic models cho Structured Output
│       └── llm.py         # Cấu hình model và logic Fallback
├── data/                  # Thư mục chứa tài liệu học thuật thô
├── storage/               # Lưu trữ dữ liệu ChromaDB cục bộ
├── app.py                 # Giao diện người dùng Streamlit
└── .env                   # Tệp cấu hình biến môi trường (Đã bỏ qua trong Git)

-------------------------------------------------------------------------------
Liên hệ: doanthang3004.work@gmail.com