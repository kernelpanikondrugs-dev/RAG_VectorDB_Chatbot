import os
import re
import json
import sqlite3
import requests
from typing import List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Import local modules
from vector_search import LocalVectorSearch
from groq_rotator import GroqKeyPool

app = FastAPI(
    title="DebtWinner RAG & GHL Automation Bridge",
    version="3.0.0",
    description="Attorney-managed financial recovery RAG chatbot with deterministic vector gating and direct GHL Inbox synchronization."
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------------
# NLP QUERY EXPANSION & REWRITING
# -------------------------------------------------------------------------
def expand_user_query(raw_query: str, groq_pool) -> str:
    """Uses Groq to translate colloquial natural language into a targeted search query for vector matching."""
    system_prompt = (
        "You are an AI search assistant for DebtWinner (attorney-managed financial recovery under VDM Attorneys).\n"
        "Your task is to take a user's natural, messy conversational question and rewrite it into a clear, concise question "
        "that matches financial recovery and debt topics (e.g., debt review distinction, fees, car/home loans, bank changes, creditors).\n"
        "Return ONLY the rewritten query text with no extra conversational filler."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User's messy query: {raw_query}"}
    ]
    try:
        completion, _ = groq_pool.chat_completion_with_fallback(messages=messages, temperature=0.0)
        rewritten = completion.choices[0].message.content.strip()
        print(f"🤖 NLP Query Rewrite: '{raw_query}' ➔ '{rewritten}'")
        return rewritten if rewritten else raw_query
    except Exception as e:
        print(f"⚠️ Query expansion failed, falling back to raw query: {e}")
        return raw_query

# -------------------------------------------------------------------------
# 1. DATABASE & VECTOR DB SETUP
# -------------------------------------------------------------------------
DB_NAME = "chatbot_audit.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_query TEXT,
            retrieved_context TEXT,
            llm_response TEXT,
            judge_score REAL,
            model_used TEXT,
            contact_name TEXT,
            contact_phone TEXT,
            contact_email TEXT,
            ghl_contact_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

vector_db = LocalVectorSearch(kb_path="kb_pairs.json")

# -------------------------------------------------------------------------
# 2. GROQ KEY POOL INITIALIZATION
# -------------------------------------------------------------------------
raw_keys = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
api_key_list = [k.strip() for k in raw_keys.split(",") if k.strip()]
groq_pool = GroqKeyPool(api_keys=api_key_list)

# -------------------------------------------------------------------------
# 3. GHL CRM & CONVERSATIONS INTEGRATION
# -------------------------------------------------------------------------
GHL_API_BASE = "https://services.leadconnectorhq.com"

def sync_ghl_contact(contact_name: Optional[str], phone: Optional[str], email: Optional[str], location_id: str, ghl_token: str) -> Optional[str]:
    first_name = "Website"
    last_name = "Visitor"
    if contact_name:
        parts = contact_name.strip().split(" ", 1)
        first_name = parts[0]
        last_name = parts[1] if len(parts) > 1 else ""

    payload = {
        "locationId": location_id,
        "firstName": first_name,
        "lastName": last_name,
        "email": email or "",
        "phone": phone or "",
        "source": "DebtWinner AI Chatbot"
    }

    headers = {
        "Authorization": f"Bearer {ghl_token}",
        "Content-Type": "application/json",
        "Version": "2021-07-28"
    }

    try:
        response = requests.post(f"{GHL_API_BASE}/contacts/upsert", json=payload, headers=headers)
        if response.status_code in [200, 201]:
            data = response.json()
            contact_obj = data.get("contact", {})
            return contact_obj.get("id") or data.get("id")
        else:
            print(f"⚠️ GHL Contact Upsert Error [{response.status_code}]: {response.text}")
    except Exception as e:
        print(f"⚠️ Exception upserting GHL contact: {e}")
        
    return None

def push_to_ghl_conversations(contact_id: str, message_body: str, direction: str, location_id: str, ghl_token: str):
    endpoint = f"{GHL_API_BASE}/conversations/messages"
    payload = {
        "type": "Live_Chat",
        "contactId": contact_id,
        "message": message_body,
        "locationId": location_id,
        "direction": direction
    }

    headers = {
        "Authorization": f"Bearer {ghl_token}",
        "Content-Type": "application/json",
        "Version": "2021-07-28"
    }

    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        if response.status_code not in [200, 201]:
            print(f"⚠️ Failed to push message to GHL Conversations [{response.status_code}]: {response.text}")
    except Exception as e:
        print(f"⚠️ Exception posting to GHL Conversations: {e}")

def trigger_ghl_handoff(contact_id: str, location_id: str, ghl_token: str):
    endpoint = f"{GHL_API_BASE}/contacts/{contact_id}/tags"
    payload = {
        "tags": ["Human Handoff Required", "AI Chatbot Escalation"]
    }
    headers = {
        "Authorization": f"Bearer {ghl_token}",
        "Content-Type": "application/json",
        "Version": "2021-07-28"
    }

    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        if response.status_code in [200, 201]:
            print(f"✅ Successfully triggered GHL human hand-off tag for contact {contact_id}")
        else:
            print(f"⚠️ Failed to add hand-off tag to GHL contact [{response.status_code}]: {response.text}")
    except Exception as e:
        print(f"⚠️ Exception triggering GHL hand-off: {e}")

# -------------------------------------------------------------------------
# 4. UTILITIES (Phone Sanitization)
# -------------------------------------------------------------------------
def sanitize_za_phone(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("27") and len(digits) == 11:
        return f"+{digits}"
    elif digits.startswith("0") and len(digits) == 10:
        return f"+27{digits[1:]}"
    return phone

# -------------------------------------------------------------------------
# 5. PYDANTIC MODELS
# -------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    user_query: str
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    judge_score: float
    sources: List[dict]
    needs_human_handoff: bool
    disclaimer: str

# -------------------------------------------------------------------------
# 6. FASTAPI ROUTES
# -------------------------------------------------------------------------
@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {
        "status": "online",
        "service": "DebtWinner RAG & GHL Automation Bridge",
        "version": "3.0.0"
    }

@app.post("/api/chat", response_model=ChatResponse)
def handle_chat(payload: ChatRequest):
    try:
        sanitized_phone = sanitize_za_phone(payload.contact_phone)

        # 0. NLP Query Expansion / Rewriting
        optimized_query = expand_user_query(payload.user_query, groq_pool)

        # 1. Retrieve knowledge chunks via vector search using the optimized query
        search_results = vector_db.search(optimized_query, top_k=3)
        if not search_results:
            search_results = []

        top_score = search_results[0].get('score', 0.0) if search_results else 0.0
        print(f"🔍 Optimized Query: '{optimized_query}' | 🎯 Top Similarity Score: {top_score:.4f}")

        # Keep threshold at 0.20 or 0.25 to allow semantic breathing room
        SIMILARITY_THRESHOLD = 0.22
        valid_chunks = [res for res in search_results if res.get('score', 0.0) >= SIMILARITY_THRESHOLD]

        if not valid_chunks:
            answer_text = (
                "I'm sorry, but I can only answer questions related to DebtWinner's financial recovery services "
                "based on our official knowledge base. Let me connect you with a human specialist who can assist you further."
            )
            judge_score = float(top_score)
            needs_human_handoff = True
            model_used = "deterministic-shortcircuit"
            context_str = "BELOW_SIMILARITY_THRESHOLD"
            sources = []
        else:
            context_str = "\n\n".join([f"- {res['content']}" for res in valid_chunks])

            system_prompt = (
                "You are the official AI assistant for DebtWinner, an attorney-managed financial recovery "
                "service under VDM Attorneys.\n\n"
                "CRITICAL RULES:\n"
                "- Answer EXCLUSIVELY using the provided Retrieved Knowledge.\n"
                "- Do not extrapolate or introduce outside facts.\n"
                "- Strictly distinguish DebtWinner from statutory debt review."
            )

            user_prompt = f"Retrieved Knowledge:\n{context_str}\n\nUser Query: {payload.user_query}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Fixed: Using chat_completion_with_fallback
            completion, model_used = groq_pool.chat_completion_with_fallback(messages=messages, temperature=0.0)
            answer_text = completion.choices[0].message.content

            judge_score = float(top_score)
            needs_human_handoff = False
            sources = [res.get("metadata", {}) for res in valid_chunks]

        # 3. GoHighLevel Contact Upsert, Conversations Sync & Hand-off Trigger
        ghl_contact_id = None
        GHL_TOKEN = os.getenv("GHL_API_TOKEN", "")
        GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID", "")

        if GHL_TOKEN and GHL_LOCATION_ID:
            try:
                ghl_contact_id = sync_ghl_contact(
                    contact_name=payload.contact_name,
                    phone=sanitized_phone,
                    email=payload.contact_email,
                    location_id=GHL_LOCATION_ID,
                    ghl_token=GHL_TOKEN
                )

                if ghl_contact_id:
                    push_to_ghl_conversations(ghl_contact_id, payload.user_query, "inbound", GHL_LOCATION_ID, GHL_TOKEN)
                    push_to_ghl_conversations(ghl_contact_id, answer_text, "outbound", GHL_LOCATION_ID, GHL_TOKEN)

                    if needs_human_handoff:
                        trigger_ghl_handoff(ghl_contact_id, GHL_LOCATION_ID, GHL_TOKEN)
            except Exception as e:
                print(f"⚠️ GHL Integration Workflow Error: {e}")

        # 4. SQLite Audit Logging
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (
                session_id, user_query, retrieved_context, llm_response, 
                judge_score, model_used, contact_name, contact_phone, contact_email, ghl_contact_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload.session_id, payload.user_query, context_str, answer_text,
            judge_score, model_used, payload.contact_name, sanitized_phone, payload.contact_email, ghl_contact_id
        ))
        conn.commit()
        conn.close()

        disclaimer = (
            "Disclaimer: DebtWinner is managed by VDM Attorneys providing financial recovery solutions. "
            "This service is strictly distinct from statutory debt review."
        )

        return ChatResponse(
            response=answer_text,
            judge_score=judge_score,
            sources=sources,
            needs_human_handoff=needs_human_handoff,
            disclaimer=disclaimer
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )