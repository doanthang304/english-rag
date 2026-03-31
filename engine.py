from __future__ import annotations

import csv
import json
import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

import fitz
import pdfplumber
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

load_dotenv()


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].tolist()


class PlanSchema(BaseModel):
    task_type: Literal["lesson_plan", "exam", "worksheet", "rubric", "general"] = Field(
        default="general"
    )
    grade: str = Field(default="unknown")
    subject: str = Field(default="Tiếng Anh")
    curriculum: str = Field(default="ALL")
    unit: str = Field(default="")
    difficulty: str = Field(default="medium")
    intent_summary: str = Field(default="")
    target_skills: list[str] = Field(default_factory=list)
    bloom_levels: list[str] = Field(default_factory=list)
    output_format: str = Field(default="markdown")
    constraints: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)

    @field_validator("task_type", mode="before")
    @classmethod
    def normalize_task_type(cls, value: Any) -> str:
        text = str(value).strip().lower()
        mapping = {
            "lesson_plan": "lesson_plan",
            "giao an": "lesson_plan",
            "giáo án": "lesson_plan",
            "lesson": "lesson_plan",
            "exam": "exam",
            "de thi": "exam",
            "đề thi": "exam",
            "kiem tra": "exam",
            "kiểm tra": "exam",
            "test": "exam",
            "worksheet": "worksheet",
            "rubric": "rubric",
            "general": "general",
        }
        return mapping.get(text, "general")

    @field_validator("grade", mode="before")
    @classmethod
    def normalize_grade(cls, value: Any) -> str:
        text = str(value).strip()
        range_match = re.search(r"(1[0-2]|[1-9])\s*[-–]\s*(1[0-2]|[1-9])", text)
        if range_match:
            return f"{range_match.group(1)}-{range_match.group(2)}"
        match = re.search(r"(1[0-2]|[1-9])", text)
        return match.group(1) if match else text

    @field_validator("curriculum", mode="before")
    @classmethod
    def normalize_curriculum(cls, value: Any) -> str:
        text = str(value).strip().upper()
        mapping = {
            "KNTT": "KNTT",
            "KẾT NỐI TRI THỨC": "KNTT",
            "KET NOI TRI THUC": "KNTT",
            "CTST": "CTST",
            "CHÂN TRỜI SÁNG TẠO": "CTST",
            "CHAN TROI SANG TAO": "CTST",
            "CĐ": "CĐ",
            "CD": "CĐ",
            "CÁNH DIỀU": "CĐ",
            "CANH DIEU": "CĐ",
            "ALL": "ALL",
        }
        mapping["CD"] = "CD"
        mapping["CÄ"] = "CD"
        mapping["CANH DIEU"] = "CD"
        return mapping.get(text, "ALL")

    @field_validator("unit", mode="before")
    @classmethod
    def normalize_unit(cls, value: Any) -> str:
        text = str(value).strip()
        range_match = re.search(r"([0-9]+)\s*[-–]\s*([0-9]+)", text)
        if range_match:
            return f"{range_match.group(1)}-{range_match.group(2)}"
        match = re.search(r"([0-9]+)", text)
        return match.group(1) if match else text

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, value: Any) -> str:
        text = str(value).strip().lower()
        mapping = {
            "easy": "easy",
            "dễ": "easy",
            "de": "easy",
            "basic": "easy",
            "medium": "medium",
            "trung bình": "medium",
            "trung binh": "medium",
            "standard": "medium",
            "hard": "hard",
            "khó": "hard",
            "kho": "hard",
            "advanced": "hard",
        }
        return mapping.get(text, "medium")

    @field_validator(
        "target_skills",
        "bloom_levels",
        "constraints",
        "search_queries",
        mode="before",
    )
    @classmethod
    def ensure_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            if not value.strip():
                return []
            return [part.strip() for part in re.split(r"[,;\n]", value) if part.strip()]
        return [str(value).strip()]


class ValidationSchema(BaseModel):
    verdict: Literal["approved", "revise"] = Field(default="approved")
    issues: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    bloom_distribution: dict[str, str] = Field(default_factory=dict)
    answer_key_quality: str = Field(default="unknown")

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, value: Any) -> str:
        text = str(value).strip().lower()
        if any(token in text for token in ["approved", "đạt", "dat", "ok", "pass"]):
            return "approved"
        if any(token in text for token in ["revise", "sửa", "sua", "chưa", "chua"]):
            return "revise"
        return "approved"

    @field_validator("issues", "improvements", mode="before")
    @classmethod
    def ensure_text_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            if not value.strip():
                return []
            return [part.strip() for part in re.split(r"[,;\n]", value) if part.strip()]
        return [str(value).strip()]

    @field_validator("bloom_distribution", mode="before")
    @classmethod
    def normalize_distribution(cls, value: Any) -> dict[str, str]:
        if isinstance(value, dict):
            return {str(key): str(item) for key, item in value.items()}
        return {}

    @field_validator("answer_key_quality", mode="before")
    @classmethod
    def normalize_answer_quality(cls, value: Any) -> str:
        return str(value).strip()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    planner_output: dict[str, Any]
    local_context: str
    web_context: str
    draft_answer: str
    review_output: dict[str, Any]
    final_answer: str


class EducationAgenticRAG:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent
        self.knowledge_dir = Path(
            os.getenv("KNOWLEDGE_DIR", self.project_root / "data" / "knowledge_base")
        )
        self.question_bank_dir = Path(
            os.getenv("QUESTION_BANK_DIR", self.project_root / "data" / "question_bank")
        )
        self.chroma_dir = Path(os.getenv("CHROMA_DIR", self.project_root / ".chroma"))
        self.collection_name = os.getenv("CHROMA_COLLECTION", "english_langgraph")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.default_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.fallback_model = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
        self._embeddings: LocalSentenceTransformerEmbeddings | None = None
        self.vectorstore: Chroma | None = None
        self.retriever = None
        self.memory = MemorySaver()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.question_bank_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    def _english_scope(self) -> str:
        return (
            "Hệ thống này chỉ phục vụ môn Tiếng Anh từ lớp 1 đến lớp 12. "
            "Không được chuyển sang môn khác. "
            "Nếu yêu cầu mơ hồ, vẫn phải suy luận trong phạm vi môn Tiếng Anh."
        )

    def _get_embeddings(self) -> LocalSentenceTransformerEmbeddings:
        if self._embeddings is None:
            self._embeddings = LocalSentenceTransformerEmbeddings(self.embedding_model_name)
        return self._embeddings

    def _get_llm(self, model_name: str | None = None, temperature: float = 0.2) -> ChatGroq:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Thiếu GROQ_API_KEY trong file .env")
        return ChatGroq(
            model=model_name or self.default_model,
            groq_api_key=api_key,
            temperature=temperature,
        )

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        parts: list[str] = []
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    parts.append(page.get_text("text"))
        except Exception:
            parts = []

        if parts:
            return "\n".join(parts)

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts)

    def _extract_tag_metadata(self, raw_tag_block: str) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for key, value in re.findall(r"\[([A-Z_]+):([^\]]+)\]", raw_tag_block):
            metadata[key] = value.strip()
        return metadata

    def _load_tagged_markdown_documents(self, text: str, path: Path) -> list[Document]:
        pattern = re.compile(r"<!--\s*((?:\[[A-Z_]+:[^\]]+\]\s*)+)-->", re.MULTILINE)
        matches = list(pattern.finditer(text))
        if not matches:
            return []

        documents: list[Document] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if not section_text:
                continue
            tags = self._extract_tag_metadata(match.group(1))
            sub_sections = re.split(r"(?=^###\s+)", section_text, flags=re.MULTILINE)
            for sub_section in sub_sections:
                sub_section = sub_section.strip()
                if not sub_section:
                    continue
                heading_line = sub_section.splitlines()[0] if sub_section.splitlines() else ""
                normalized_grade = self._normalize_grade_value(
                    tags.get("GRADE", "") or self._infer_grade(sub_section, heading_line)
                )
                normalized_unit = self._normalize_unit_value(
                    tags.get("UNIT", "") or self._extract_unit_from_text(heading_line)
                )
                documents.append(
                    Document(
                        page_content=sub_section,
                        metadata={
                            "source_name": path.name,
                            "source_path": str(path),
                            "doc_type": "curriculum_tagged",
                            "grade": normalized_grade,
                            "curriculum": self._normalize_curriculum_value(tags.get("CURRICULUM", "ALL")),
                            "unit": normalized_unit,
                            "topic": tags.get("TOPIC", heading_line.replace("###", "").strip()),
                            "content_type": tags.get("TYPE", ""),
                            "title": heading_line.replace("###", "").strip() or tags.get("TOPIC", path.stem),
                            "subject": "Tiếng Anh",
                        },
                    )
                )
        return documents

    def _extract_unit_from_text(self, text: str) -> str:
        range_match = re.search(
            r"unit[s]?\s*([0-9]+)\s*[-–]\s*([0-9]+)",
            str(text),
            re.IGNORECASE,
        )
        if range_match:
            return f"{range_match.group(1)}-{range_match.group(2)}"
        match = re.search(r"unit[s]?\s*([0-9]+)", str(text), re.IGNORECASE)
        return match.group(1) if match else ""

    def _normalize_grade_value(self, value: str) -> str:
        text = str(value).strip().upper()
        if not text:
            return "ALL"
        if text == "ALL":
            return "ALL"
        range_match = re.search(r"(1[0-2]|[1-9])\s*[-–]\s*(1[0-2]|[1-9])", text)
        if range_match:
            return f"{range_match.group(1)}-{range_match.group(2)}"
        match = re.search(r"(1[0-2]|[1-9])", text)
        return match.group(1) if match else text

    def _normalize_curriculum_value(self, value: str) -> str:
        text = str(value).strip().upper()
        mapping = {
            "KNTT": "KNTT",
            "CTST": "CTST",
            "CĐ": "CĐ",
            "CD": "CĐ",
            "ALL": "ALL",
        }
        mapping["CD"] = "CD"
        mapping["CÄ"] = "CD"
        return mapping.get(text, text or "ALL")

    def _normalize_unit_value(self, value: str) -> str:
        text = str(value).strip()
        range_match = re.search(r"([0-9]+)\s*[-–]\s*([0-9]+)", text)
        if range_match:
            return f"{range_match.group(1)}-{range_match.group(2)}"
        match = re.search(r"([0-9]+)", text)
        return match.group(1) if match else text

    def _matches_numeric_scope(self, requested: str, actual: str) -> bool:
        requested_text = str(requested).strip().upper()
        actual_text = str(actual).strip().upper()
        if not requested_text or requested_text == "ALL":
            return True
        if not actual_text or actual_text == "ALL":
            return False
        if requested_text == actual_text:
            return True

        def expand(value: str) -> set[int]:
            range_match = re.fullmatch(r"([0-9]+)\s*[-–]\s*([0-9]+)", value)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                low, high = sorted((start, end))
                return set(range(low, high + 1))
            if value.isdigit():
                return {int(value)}
            return set()

        requested_values = expand(requested_text)
        actual_values = expand(actual_text)
        if not requested_values or not actual_values:
            return requested_text == actual_text
        return bool(requested_values & actual_values)

    def _question_row_to_document(self, row: dict[str, Any], source_path: Path) -> Document:
        options = row.get("options", "")
        if isinstance(options, list):
            options_text = "\n".join(f"- {option}" for option in options)
        else:
            options_text = str(options)

        content = (
            f"Question: {row.get('question', '')}\n"
            f"Options:\n{options_text}\n"
            f"Answer: {row.get('answer', '')}\n"
            f"Explanation: {row.get('explanation', '')}\n"
            f"Grade: {row.get('grade', '')}\n"
            f"Skill: {row.get('skill', '')}\n"
            f"Level: {row.get('level', '')}\n"
            f"Topic: {row.get('topic', '')}"
        )
        return Document(
            page_content=content,
            metadata={
                "source_name": source_path.name,
                "source_path": str(source_path),
                "doc_type": "question_bank",
                "grade": self._normalize_grade_value(str(row.get("grade", "all"))),
                "curriculum": self._normalize_curriculum_value(str(row.get("curriculum", "ALL"))),
                "unit": self._normalize_unit_value(str(row.get("unit", ""))),
                "topic": str(row.get("topic", "")),
                "difficulty": str(row.get("difficulty", "medium")),
                "content_type": "QUESTION_BANK",
                "skills": row.get("skill", "Integrated"),
                "title": row.get("topic", "Question Bank"),
                "subject": "Tiếng Anh",
            },
        )

    def _load_documents(self) -> list[Document]:
        documents: list[Document] = []
        canonical_source = "knowledge_base_tieng_anh_1_12.md"
        legacy_scaffold_sources = {
            "english_assessment_templates.md",
            "english_curriculum_framework.md",
            "english_lower_secondary_grades_6_9.md",
            "english_primary_grades_1_5.md",
            "english_upper_secondary_grades_10_12.md",
        }
        canonical_exists = (self.knowledge_dir / canonical_source).exists()

        for path in sorted(self.knowledge_dir.rglob("*")):
            if not path.is_file():
                continue
            if canonical_exists and path.name in legacy_scaffold_sources:
                continue
            suffix = path.suffix.lower()
            if suffix not in {".md", ".txt", ".pdf"}:
                continue

            if suffix == ".pdf":
                text = self._extract_pdf_text(path)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")

            if not text.strip():
                continue

            if suffix in {".md", ".txt"} and "[GRADE:" in text and "[UNIT:" in text:
                documents.extend(self._load_tagged_markdown_documents(text, path))
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_name": path.name,
                        "source_path": str(path),
                        "doc_type": "pdf" if suffix == ".pdf" else "curriculum",
                        "grade": self._infer_grade(text, path.name),
                        "curriculum": "ALL",
                        "unit": "",
                        "topic": path.stem,
                        "content_type": "GENERAL",
                        "skills": "Integrated",
                        "title": path.stem,
                        "subject": "Tiếng Anh",
                    },
                )
            )

        for path in sorted(self.question_bank_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".csv":
                with path.open("r", encoding="utf-8", errors="ignore", newline="") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        documents.append(self._question_row_to_document(row, path))
            elif suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                items = payload.get("questions", payload) if isinstance(payload, dict) else payload
                if isinstance(items, list):
                    for row in items:
                        if isinstance(row, dict):
                            documents.append(self._question_row_to_document(row, path))

        return documents

    def _infer_grade(self, text: str, source_name: str) -> str:
        combined = f"{source_name}\n{text}"
        match = re.search(r"(lớp|lop|grade)\s*(1[0-2]|[1-9])", combined, re.IGNORECASE)
        if match:
            return match.group(2)
        if "1_5" in source_name:
            return "1-5"
        if "6_9" in source_name:
            return "6-9"
        if "10_12" in source_name:
            return "10-12"
        return "all"

    def _enrich_chunk_metadata(self, documents: list[Document]) -> list[Document]:
        for doc in documents:
            content = doc.page_content
            heading_match = re.search(r"^###\s+(.+)$", content, re.MULTILINE)
            heading_line = heading_match.group(1).strip() if heading_match else ""
            skills = []
            for keyword in [
                "Listening",
                "Speaking",
                "Reading",
                "Writing",
                "Vocabulary",
                "Grammar",
                "Pronunciation",
                "Communication",
            ]:
                if re.search(keyword, content, re.IGNORECASE):
                    skills.append(keyword)
            if not skills:
                skills = ["Integrated"]
            doc.metadata["skills"] = " | ".join(skills)
            doc.metadata["subject"] = "Tiếng Anh"
            doc.metadata["curriculum"] = self._normalize_curriculum_value(
                str(doc.metadata.get("curriculum", "ALL"))
            )
            inferred_grade = self._normalize_grade_value(
                self._infer_grade(content, heading_line or str(doc.metadata.get("title", "")))
            )
            current_grade = self._normalize_grade_value(str(doc.metadata.get("grade", "ALL")))
            inferred_unit = self._normalize_unit_value(
                self._extract_unit_from_text(heading_line) or self._extract_unit_from_text(content)
            )
            current_unit = self._normalize_unit_value(str(doc.metadata.get("unit", "")))
            doc.metadata["grade"] = current_grade if current_grade != "ALL" else inferred_grade
            doc.metadata["unit"] = current_unit or inferred_unit
            doc.metadata["difficulty"] = str(doc.metadata.get("difficulty", "medium"))
            doc.metadata["topic"] = str(
                doc.metadata.get("topic")
                or heading_line
                or doc.metadata.get("title", "")
            )
            doc.metadata["content_type"] = str(doc.metadata.get("content_type", "GENERAL"))
            doc.metadata["heading"] = heading_line
            doc.metadata["chunk_id"] = str(uuid4())
        return documents

    def ingest_documents(self) -> int:
        if self.chroma_dir.exists():
            shutil.rmtree(self.chroma_dir, ignore_errors=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        raw_documents = self._load_documents()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=180,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_documents)
        chunks = self._enrich_chunk_metadata(chunks)

        if not chunks:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=str(self.chroma_dir),
                embedding_function=self._get_embeddings(),
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
            return 0

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._get_embeddings(),
            collection_name=self.collection_name,
            persist_directory=str(self.chroma_dir),
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
        return len(chunks)

    def ensure_vectorstore(self) -> None:
        if self.vectorstore is not None and self.retriever is not None:
            return
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.chroma_dir),
            embedding_function=self._get_embeddings(),
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})

    def _metadata_match_bonus(self, metadata: dict[str, Any], plan: dict[str, Any]) -> float:
        bonus = 0.0
        grade = str(plan.get("grade", "")).strip()
        curriculum = self._normalize_curriculum_value(str(plan.get("curriculum", "ALL")))
        unit = str(plan.get("unit", "")).strip()
        metadata_grade = self._normalize_grade_value(str(metadata.get("grade", "ALL")))
        metadata_curriculum = self._normalize_curriculum_value(str(metadata.get("curriculum", "ALL")))
        metadata_unit = self._normalize_unit_value(str(metadata.get("unit", "")))
        doc_type = str(metadata.get("doc_type", "")).lower()
        source_name = str(metadata.get("source_name", "")).lower()
        task_type = str(plan.get("task_type", "")).strip().lower()

        if grade:
            if metadata_grade == grade:
                bonus += 3.5
            elif self._matches_numeric_scope(grade, metadata_grade):
                bonus += 2.2
            elif metadata_grade not in {"", "ALL"}:
                bonus -= 4.5
                if unit:
                    bonus -= 2.0
        if curriculum and curriculum != "ALL":
            if metadata_curriculum == curriculum:
                bonus += 2.5
            elif metadata_curriculum not in {"", "ALL"}:
                bonus -= 2.0
        if unit:
            if metadata_unit == unit:
                bonus += 4.0
            elif self._matches_numeric_scope(unit, metadata_unit):
                bonus += 3.0
            elif metadata_unit:
                bonus -= 3.0
            else:
                bonus -= 3.5
        if unit and doc_type == "curriculum_tagged":
            bonus += 1.2
        if unit and doc_type == "question_bank" and not metadata_unit:
            bonus -= 1.0
        if task_type == "lesson_plan" and doc_type == "curriculum_tagged":
            bonus += 0.8
        if task_type == "exam" and doc_type == "question_bank":
            bonus += 0.8
        if "knowledge_base_tieng_anh_1_12" in source_name:
            bonus += 2.0
        return bonus

    def _retrieve_ranked_documents(
        self,
        queries: list[str],
        plan: dict[str, Any],
        top_k: int = 6,
    ) -> list[Document]:
        self.ensure_vectorstore()
        ranked: list[tuple[float, Document]] = []
        seen: set[tuple[str, str]] = set()
        grade = str(plan.get("grade", "")).strip()
        curriculum = self._normalize_curriculum_value(str(plan.get("curriculum", "ALL")))
        unit = str(plan.get("unit", "")).strip()
        target_skills = " ".join(plan.get("target_skills", []))
        query_candidates = [query for query in queries[:3] if query]
        if grade or unit or curriculum != "ALL" or target_skills:
            precise_query = " ".join(
                part
                for part in [
                    "Tieng Anh",
                    f"lop {grade}" if grade else "",
                    f"bo {curriculum}" if curriculum and curriculum != "ALL" else "",
                    f"unit {unit}" if unit else "",
                    target_skills,
                ]
                if part
            )
            if precise_query:
                query_candidates.insert(0, precise_query)

        for query in query_candidates[:4]:
            filter_chain: list[dict[str, Any] | None] = []
            if grade:
                filter_chain.append({"$and": [{"grade": grade}, {"source_name": "knowledge_base_tieng_anh_1_12.md"}]})
                filter_chain.append({"grade": grade})
            if curriculum and curriculum != "ALL":
                filter_chain.append({"curriculum": curriculum})
            filter_chain.append({"source_name": "knowledge_base_tieng_anh_1_12.md"})
            filter_chain.append(None)

            for metadata_filter in filter_chain:
                try:
                    hits = self.vectorstore.similarity_search_with_score(
                        query,
                        k=10,
                        filter=metadata_filter,
                    )
                except Exception:
                    continue
                for doc, score in hits:
                    key = (
                        str(doc.metadata.get("source_name", "")),
                        str(doc.metadata.get("title", "")),
                        str(doc.metadata.get("grade", "")),
                        str(doc.metadata.get("curriculum", "")),
                        str(doc.metadata.get("unit", "")),
                        str(doc.metadata.get("content_type", "")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    adjusted_score = float(score) - self._metadata_match_bonus(doc.metadata, plan)
                    ranked.append((adjusted_score, doc))
        ranked.sort(key=lambda item: item[0])
        ordered_docs = [doc for _, doc in ranked]
        if grade:
            grade_filtered_docs = [
                doc
                for doc in ordered_docs
                if self._matches_numeric_scope(
                    grade,
                    self._normalize_grade_value(str(doc.metadata.get("grade", "ALL"))),
                )
                or self._normalize_grade_value(str(doc.metadata.get("grade", "ALL"))) in {"", "ALL"}
            ]
            if grade_filtered_docs:
                ordered_docs = grade_filtered_docs
        if unit:
            unit_priority_docs = []
            support_docs = []
            for doc in ordered_docs:
                doc_unit = self._normalize_unit_value(str(doc.metadata.get("unit", "")))
                if doc_unit and (
                    doc_unit == unit or self._matches_numeric_scope(unit, doc_unit)
                ):
                    unit_priority_docs.append(doc)
                else:
                    support_docs.append(doc)
            ordered_docs = unit_priority_docs + support_docs
        return ordered_docs[:top_k]

    def _web_search(self, query: str, grade: str, enabled: bool) -> str:
        if not enabled:
            return "Tra cứu web đang tắt."
        search_query = (
            f"{query} English grade {grade} worksheet test lesson plan filetype:pdf OR site:.edu OR site:.vn"
        )
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=5))
        except Exception as exc:
            return f"Không thể tra cứu web: {exc}"
        if not results:
            return "Không tìm thấy kết quả web phù hợp."
        parts = []
        for index, item in enumerate(results, start=1):
            parts.append(
                f"[Nguồn web {index}]\n"
                f"Tiêu đề: {item.get('title', '')}\n"
                f"Tóm tắt: {item.get('body', '')}\n"
                f"Liên kết: {item.get('href', '')}"
            )
        return "\n\n".join(parts)

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _coerce_validation_from_text(self, text: str) -> dict[str, Any]:
        verdict = "approved"
        lowered = text.lower()
        if any(token in lowered for token in ["revise", "sửa", "sua", "chưa", "chua"]):
            verdict = "revise"

        issues = re.findall(r"[-*]\s+(.+)", text)
        issues = issues[:3]
        improvements = issues[:]

        bloom_distribution: dict[str, str] = {}
        for key, value in re.findall(
            r"(Remember|Understand|Apply|Analyze|Evaluate|Create|Nhận biết|Thông hiểu|Vận dụng|Phân tích|Đánh giá|Tạo lập)\s*[:\-]\s*([0-9]+%?)",
            text,
            re.IGNORECASE,
        ):
            bloom_distribution[key] = value

        answer_key_quality = "unknown"
        if "đáp án" in lowered or "answer key" in lowered:
            answer_key_quality = "present"

        return {
            "verdict": verdict,
            "issues": issues,
            "improvements": improvements,
            "bloom_distribution": bloom_distribution,
            "answer_key_quality": answer_key_quality,
        }

    def _planner_node(self, model_name: str):
        parser = JsonOutputParser(pydantic_object=PlanSchema)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{self._english_scope()} "
                    "Bạn là planner. Hãy phân tích yêu cầu và trả về kế hoạch JSON chi tiết cho giáo án, đề thi, worksheet hoặc rubric. "
                    "Phải xác định thật rõ: khối lớp, bộ sách (KNTT/CĐ/CTST/ALL), unit, độ khó, kỹ năng mục tiêu và ràng buộc đầu ra. "
                    "Nếu người dùng đã nêu lớp, bộ sách, unit hoặc độ khó thì không được suy diễn khác đi.",
                ),
                (
                    "human",
                    "Yêu cầu người dùng:\n{query}\n\n"
                    "Trả về JSON với các trường bắt buộc gồm task_type, grade, subject, curriculum, unit, difficulty, "
                    "intent_summary, target_skills, bloom_levels, output_format, constraints, search_queries.\n\n"
                    "Hướng dẫn định dạng:\n{format_instructions}",
                ),
            ]
        )
        chain = prompt | self._get_llm(model_name, temperature=0.1) | parser

        def planner(state: AgentState) -> dict[str, Any]:
            raw = chain.invoke(
                {
                    "query": state["user_query"],
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            plan = PlanSchema(**raw).model_dump()
            query_lower = state["user_query"].lower()
            query_lower = query_lower.replace("\u0111", "d")
            query_ascii = "".join(
                char
                for char in unicodedata.normalize("NFKD", query_lower.replace("đ", "d"))
                if not unicodedata.combining(char)
            )
            if plan["task_type"] == "general":
                if any(token in query_lower for token in ["đề", "de ", "kiểm tra", "kiem tra", "exam", "test"]):
                    plan["task_type"] = "exam"
                elif any(token in query_lower for token in ["giáo án", "giao an", "lesson plan"]):
                    plan["task_type"] = "lesson_plan"
                elif "worksheet" in query_lower:
                    plan["task_type"] = "worksheet"
                elif "rubric" in query_lower:
                    plan["task_type"] = "rubric"
            if not plan.get("grade") or plan.get("grade") == "unknown":
                grade_match = re.search(r"(lớp|lop|grade)\s*(1[0-2]|[1-9])", query_lower, re.IGNORECASE)
                if grade_match:
                    plan["grade"] = grade_match.group(2)
            if not plan.get("unit"):
                unit_match = re.search(r"unit\s*([0-9]+)", query_lower, re.IGNORECASE)
                if unit_match:
                    plan["unit"] = unit_match.group(1)
            if plan.get("curriculum", "ALL") == "ALL":
                if "kntt" in query_lower or "kết nối tri thức" in query_lower or "ket noi tri thuc" in query_lower:
                    plan["curriculum"] = "KNTT"
                elif "ctst" in query_lower or "chân trời sáng tạo" in query_lower or "chan troi sang tao" in query_lower:
                    plan["curriculum"] = "CTST"
                elif "cánh diều" in query_lower or "canh dieu" in query_lower or re.search(r"\bcđ\b", query_lower):
                    plan["curriculum"] = "CĐ"
            if plan.get("difficulty", "medium") == "medium":
                if any(token in query_lower for token in ["trung bình", "trung binh", "medium", "vừa sức", "vua suc"]):
                    plan["difficulty"] = "medium"
                elif any(token in query_lower for token in ["dễ", " de ", "cơ bản", "co ban", "basic", "easy"]):
                    plan["difficulty"] = "easy"
                elif any(token in query_lower for token in ["khó", "kho", "nâng cao", "nang cao", "advanced", "hard"]):
                    plan["difficulty"] = "hard"
            grade_match = re.search(r"(lá»›p|lop|grade)\s*(1[0-2]|[1-9])", query_lower, re.IGNORECASE)
            if grade_match:
                plan["grade"] = grade_match.group(2)
            unit_match = re.search(
                r"unit[s]?\s*([0-9]+(?:\s*[-–]\s*[0-9]+)?)",
                query_lower,
                re.IGNORECASE,
            )
            if unit_match:
                plan["unit"] = self._normalize_unit_value(unit_match.group(1))
            if "kntt" in query_lower or "káº¿t ná»‘i tri thá»©c" in query_lower or "ket noi tri thuc" in query_lower:
                plan["curriculum"] = "KNTT"
            elif "ctst" in query_lower or "chĂ¢n trá»i sĂ¡ng táº¡o" in query_lower or "chan troi sang tao" in query_lower:
                plan["curriculum"] = "CTST"
            elif "cĂ¡nh diá»u" in query_lower or "canh dieu" in query_lower or re.search(r"\bcÄ‘\b", query_lower):
                plan["curriculum"] = "CÄ"
            explicit_difficulty = None
            if any(token in query_lower for token in ["trung bĂ¬nh", "trung binh", "medium", "vá»«a sá»©c", "vua suc"]):
                explicit_difficulty = "medium"
            elif any(token in query_lower for token in ["dá»…", " de ", "cÆ¡ báº£n", "co ban", "basic", "easy"]):
                explicit_difficulty = "easy"
            elif any(token in query_lower for token in ["khĂ³", "kho", "nĂ¢ng cao", "nang cao", "advanced", "hard"]):
                explicit_difficulty = "hard"
            if explicit_difficulty:
                plan["difficulty"] = explicit_difficulty
            if any(token in query_ascii for token in ["giao an", "lesson plan"]):
                plan["task_type"] = "lesson_plan"
            elif any(token in query_ascii for token in ["de thi", "kiem tra", "exam", "test"]):
                plan["task_type"] = "exam"
            elif "worksheet" in query_ascii:
                plan["task_type"] = "worksheet"
            elif "rubric" in query_ascii:
                plan["task_type"] = "rubric"
            grade_match_ascii = re.search(r"(lop|grade)\s*(1[0-2]|[1-9])", query_ascii, re.IGNORECASE)
            if grade_match_ascii:
                plan["grade"] = grade_match_ascii.group(2)
            unit_match_ascii = re.search(
                r"unit[s]?\s*([0-9]+(?:\s*[-–]\s*[0-9]+)?)",
                query_ascii,
                re.IGNORECASE,
            )
            if unit_match_ascii:
                plan["unit"] = self._normalize_unit_value(unit_match_ascii.group(1))
            if "kntt" in query_ascii or "ket noi tri thuc" in query_ascii:
                plan["curriculum"] = "KNTT"
            elif "ctst" in query_ascii or "chan troi sang tao" in query_ascii:
                plan["curriculum"] = "CTST"
            elif "canh dieu" in query_ascii or re.search(r"\bcd\b", query_ascii):
                plan["curriculum"] = "CD"
            explicit_difficulty_ascii = None
            if any(token in query_ascii for token in ["trung binh", "medium", "vua suc"]):
                explicit_difficulty_ascii = "medium"
            elif any(token in query_ascii for token in [" de ", "co ban", "basic", "easy"]):
                explicit_difficulty_ascii = "easy"
            elif any(token in query_ascii for token in ["kho", "nang cao", "advanced", "hard"]):
                explicit_difficulty_ascii = "hard"
            if explicit_difficulty_ascii:
                plan["difficulty"] = explicit_difficulty_ascii
            if not plan.get("target_skills"):
                inferred_skills = []
                for keyword in ["reading", "listening", "speaking", "writing", "grammar", "vocabulary", "pronunciation"]:
                    if keyword in query_lower:
                        inferred_skills.append(keyword.capitalize())
                if inferred_skills:
                    plan["target_skills"] = inferred_skills
            plan["subject"] = "Tiếng Anh"
            return {"planner_output": plan}

        return planner

    def _retriever_node(self, web_search_enabled: bool):
        def retrieve(state: AgentState) -> dict[str, Any]:
            plan = state.get("planner_output", {})
            queries = plan.get("search_queries") or [state["user_query"]]
            docs = self._retrieve_ranked_documents(queries, plan, top_k=6)

            local_parts = []
            for index, doc in enumerate(docs[:6], start=1):
                local_parts.append(
                    f"[Nguồn {index} | {doc.metadata.get('source_name', 'unknown')} | "
                    f"grade={doc.metadata.get('grade', '')} | curriculum={doc.metadata.get('curriculum', '')} | "
                    f"unit={doc.metadata.get('unit', '')} | skills={doc.metadata.get('skills', '')} | "
                    f"type={doc.metadata.get('content_type', '')}]\n"
                    f"{doc.page_content.strip()}"
                )
            local_context = (
                "\n\n".join(local_parts) if local_parts else "Không tìm thấy ngữ cảnh nội bộ phù hợp."
            )
            web_context = self._web_search(
                query=state["user_query"],
                grade=plan.get("grade", ""),
                enabled=web_search_enabled,
            )
            return {"local_context": local_context, "web_context": web_context}

        return retrieve

    def _generator_node(self, model_name: str):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{self._english_scope()} "
                    "Bạn là generator cho hệ thống Agentic RAG môn Tiếng Anh. "
                    "Hãy sinh nội dung mới, không sao chép máy móc từ nguồn. "
                    "Phải bám chính xác khối lớp, bộ sách, unit và độ khó đã được planner xác định. "
                    "Nếu tạo đề kiểm tra phải có đáp án và thang điểm. "
                    "Nếu tạo giáo án phải có mục tiêu, học liệu, tiến trình và đánh giá.",
                ),
                (
                    "human",
                    "Yêu cầu gốc:\n{query}\n\n"
                    "Kế hoạch:\n{plan}\n\n"
                    "Ngữ cảnh nội bộ:\n{local_context}\n\n"
                    "Ngữ cảnh tham khảo web:\n{web_context}\n\n"
                    "Hãy tạo bản nháp đầu ra rõ ràng, dùng được ngay. "
                    "Nếu thiếu dữ liệu đúng unit hoặc đúng bộ sách thì phải nói rõ giả định thay vì tự ý đổi sang unit khác.",
                ),
            ]
        )
        chain = prompt | self._get_llm(model_name, temperature=0.3) | StrOutputParser()

        def generate(state: AgentState) -> dict[str, Any]:
            answer = chain.invoke(
                {
                    "query": state["user_query"],
                    "plan": json.dumps(state["planner_output"], ensure_ascii=False, indent=2),
                    "local_context": state["local_context"],
                    "web_context": state["web_context"],
                }
            )
            return {
                "draft_answer": answer,
                "messages": [AIMessage(content="Đã tạo bản nháp nội dung.")],
            }

        return generate

    def _validator_node(self, model_name: str):
        parser = JsonOutputParser(pydantic_object=ValidationSchema)
        review_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{self._english_scope()} "
                    "Bạn là validator. Hãy kiểm tra bản nháp theo 6 tiêu chí: đúng khối lớp, đúng môn, đúng bộ sách, đúng unit, đúng độ khó, có đáp án/thang điểm khi cần, và phân phối Bloom hợp lý.",
                ),
                (
                    "human",
                    "Yêu cầu gốc:\n{query}\n\nKế hoạch:\n{plan}\n\nBản nháp:\n{draft}\n\n"
                    "BẮT BUỘC trả về JSON thuần theo đúng schema sau, không thêm markdown hay giải thích:\n{format_instructions}",
                ),
            ]
        )
        review_llm = self._get_llm(model_name, temperature=0.1)
        review_chain = review_prompt | review_llm | parser
        revise_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{self._english_scope()} "
                    "Bạn đang chỉnh sửa bản nháp theo phản hồi validator. Hãy trả về phiên bản tốt hơn, đầy đủ hơn và vẫn giữ đúng khối lớp, bộ sách, unit, độ khó.",
                ),
                (
                    "human",
                    "Yêu cầu gốc:\n{query}\n\nKế hoạch:\n{plan}\n\nNgữ cảnh nội bộ:\n{local_context}\n\n"
                    "Ngữ cảnh web:\n{web_context}\n\nBản nháp hiện tại:\n{draft}\n\n"
                    "Vấn đề:\n{issues}\n\nGợi ý cải thiện:\n{improvements}",
                ),
            ]
        )
        revise_chain = revise_prompt | self._get_llm(model_name, temperature=0.2) | StrOutputParser()

        def validator(state: AgentState) -> dict[str, Any]:
            payload = {
                "query": state["user_query"],
                "plan": json.dumps(state["planner_output"], ensure_ascii=False, indent=2),
                "draft": state["draft_answer"],
                "format_instructions": parser.get_format_instructions(),
            }
            try:
                review_raw = review_chain.invoke(payload)
            except OutputParserException as exc:
                review_text = getattr(exc, "llm_output", "") or str(exc)
                review_raw = self._extract_json_object(review_text) or self._coerce_validation_from_text(
                    review_text
                )
            review = ValidationSchema(**review_raw).model_dump()
            final_answer = state["draft_answer"]
            if review["verdict"] == "revise":
                final_answer = revise_chain.invoke(
                    {
                        "query": state["user_query"],
                        "plan": json.dumps(state["planner_output"], ensure_ascii=False, indent=2),
                        "local_context": state["local_context"],
                        "web_context": state["web_context"],
                        "draft": state["draft_answer"],
                        "issues": "\n".join(review.get("issues", [])),
                        "improvements": "\n".join(review.get("improvements", [])),
                    }
                )
            return {
                "review_output": review,
                "final_answer": final_answer,
                "messages": [AIMessage(content=final_answer)],
            }

        return validator

    def build_graph(self, model_name: str | None = None, web_search_enabled: bool = False):
        chosen_model = model_name or self.default_model
        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner_node(chosen_model))
        graph.add_node("retrieve", self._retriever_node(web_search_enabled))
        graph.add_node("generate", self._generator_node(chosen_model))
        graph.add_node("validator", self._validator_node(chosen_model))
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "validator")
        graph.add_edge("validator", END)
        return graph.compile(checkpointer=self.memory)

    def invoke(
        self,
        query: str,
        thread_id: str,
        model_name: str | None = None,
        web_search_enabled: bool = False,
    ) -> dict[str, Any]:
        if self.vectorstore is None:
            self.ingest_documents()
        app = self.build_graph(model_name, web_search_enabled)
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "planner_output": {},
            "local_context": "",
            "web_context": "",
            "draft_answer": "",
            "review_output": {},
            "final_answer": "",
        }
        config = {"configurable": {"thread_id": thread_id}}
        return app.invoke(initial_state, config=config)
