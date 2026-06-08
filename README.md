# 🦺 Compliance Vision AI — PPE Compliance Monitor

> **Always-on AI-powered workplace safety monitoring** — detects PPE violations via YOLOv8, retrieves exact regulatory clauses via RAG, and generates structured compliance reports with GPT-4o-mini.

---

## How to Run the Full Project

This project has two parts:

- `frontend/` - Next.js web app with login, dashboard, AI chatbot, and Supabase PostgreSQL data storage.
- Root Python API - FastAPI RAG backend used by the chatbot for safety answers.

### 1. Open the Project

```powershell
cd C:\Project\PPE
```

### 2. Create the Python Environment

If the virtual environment already exists, skip the first command.

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure the Python Backend

Create or edit the root `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

### 4. Build the RAG Knowledge Base

Run this once before starting the RAG API:

```powershell
python build_kb.py --force
```

### 5. Start the Python RAG API

Keep this terminal running:

```powershell
uvicorn api_server:app --reload --host 127.0.0.1 --port 8001
```

You can test it here:

```text
http://127.0.0.1:8001/docs
```

### 6. Configure the Frontend

Open a second terminal:

```powershell
cd C:\Project\PPE\frontend
```

Create or edit `frontend/.env.local` and use your Supabase PostgreSQL connection strings:

```env
DATABASE_URL="postgresql://postgres.PROJECT_REF:DB_PASSWORD@REGION.pooler.supabase.com:6543/postgres?pgbouncer=true"
DIRECT_URL="postgresql://postgres.PROJECT_REF:DB_PASSWORD@REGION.pooler.supabase.com:5432/postgres"
NEXTAUTH_SECRET=change-this-secret
NEXTAUTH_URL=http://localhost:3000
RAG_API_URL=http://127.0.0.1:8001/answer

# Optional: email violation PDF when a new notification appears
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
SMTP_FROM="Compliance Vision AI <your-email@gmail.com>"
```

### 7. Install Frontend Dependencies

```powershell
npm install
```

If PowerShell blocks `npx`, use `npx.cmd` in the commands below.

### 8. Prepare the Supabase Database

```powershell
npx.cmd prisma generate
npx.cmd prisma migrate deploy
```

This creates the app tables in Supabase: users, password reset tokens, team invites, violations, chat sessions, and chat messages.

### 9. Start the Frontend

```powershell
npm run dev
```

Open:

```text
http://localhost:3000
```

### 10. Use the App

1. Go to `http://localhost:3000/register`.
2. Create a user account.
3. Login at `http://localhost:3000/login`.
4. Click `AI` in the navbar.
5. Send messages to the chatbot.
6. Chat history is saved on the left side and stored in Supabase.

### Notes

- The chatbot still opens and saves chat history to Supabase if the Python RAG API is not running, but answers will use a fallback response.
- Keep both servers running for the full experience:
  - Frontend: `http://localhost:3000`
  - RAG API: `http://127.0.0.1:8001`
- Run checks with:

```powershell
cd C:\Project\PPE\frontend
npm run lint
npm run build
```

---

## 📁 Project Structure

```
PPE/
├── app.py                  ← Streamlit dashboard (main entry point)
├── build_kb.py             ← CLI: build FAISS knowledge base from PDFs
├── requirements.txt
├── .env.example            ← Copy to .env and add your API key
│
├── data/
│   ├── raw/                ← ⬅ Place your safety document PDFs here
│   └── processed/          ← Auto-generated: chunks.jsonl, metadata.json, index.faiss
│
├── models/
│   └── best.pt             ← ⬅ Place your YOLOv8 model here
│
├── outputs/
│   ├── screenshots/        ← Annotated violation frames (auto-saved)
│   └── reports/            ← Exported JSON reports
│
└── src/
    ├── config.py           ← All configuration (paths, thresholds, LLM settings)
    ├── extractor.py        ← PyMuPDF PDF text extraction
    ├── cleaner.py          ← Text cleaning / normalisation
    ├── chunker.py          ← Sliding-window sentence chunker
    ├── embedder.py         ← Sentence-Transformers embeddings
    ├── vector_store.py     ← FAISS index build/load + ChunkRecord dataclass
    ├── retriever.py        ← Semantic retrieval over FAISS
    ├── violation_logic.py  ← Rule-based violation detection engine
    ├── video_processor.py  ← OpenCV frame extraction + YOLO inference
    └── report_generator.py ← OpenAI LLM report generation
```

---

## 🚀 Quick Start

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Configure environment
```bash
copy .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3 — Add your files
| What | Where |
|------|-------|
| Safety document PDF(s) | `data/raw/` |
| YOLOv8 model (`best.pt`) | `models/` |

### 4 — Build the knowledge base
```bash
python build_kb.py
# Rebuild: python build_kb.py --force
```

### 5 — Launch the dashboard
```bash
streamlit run app.py
```

---

## ⚙️ End-to-End Pipeline

```
CCTV Video / Image
        │
        ▼
┌──────────────────┐
│  OpenCV Frame    │  Extract frames at configurable FPS
│  Extraction      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  YOLOv8 Object   │  Detect: person, helmet, safety_vest,
│  Detection       │  face_mask, gloves, fire_extinguisher,
└────────┬─────────┘  emergency_exit, restricted_zone_marker
         │
         ▼
┌──────────────────┐
│  Violation       │  Rule-based logic: PPE absence,
│  Logic Engine    │  restricted zones, blocked exits,
└────────┬─────────┘  overcrowding
         │
         ▼
┌──────────────────┐
│  RAG Retrieval   │  FAISS semantic search over indexed
│  (FAISS)         │  safety documents → top-3 SOP clauses
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM Report      │  GPT-4o-mini generates structured JSON
│  Generator       │  report + plain-language narrative
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Streamlit       │  Live dashboard, evidence frames,
│  Dashboard       │  report log, JSON export
└──────────────────┘
```

---

## 🎯 Detectable Violations

| Code | Description |
|------|-------------|
| `PPE_HELMET_MISSING` | Person without helmet in hard-hat zone |
| `PPE_VEST_MISSING` | Person without high-vis vest |
| `PPE_MASK_MISSING` | Person without face mask in required zone |
| `PPE_GLOVES_MISSING` | Person without gloves in hand-protection zone |
| `RESTRICTED_ZONE` | Unauthorised personnel in restricted area |
| `BLOCKED_EXIT` | Emergency exit obstructed |
| `BLOCKED_EXTINGUISHER` | Fire extinguisher occluded |
| `OVERCROWDING` | Zone occupancy exceeds threshold |
| `UNSAFE_POSTURE` | Ergonomic/fall risk detected |

---

## 📊 Output JSON Schema

```json
{
  "violation_id":         "UUID",
  "violation_type":       "PPE_HELMET_MISSING",
  "certainty":            "Confirmed | Probable | Possible",
  "confidence_score":     0.87,
  "timestamp":            "00:01:23",
  "zone":                 "general_floor",
  "camera_id":            "CAM-01",
  "evidence_frame":       "outputs/screenshots/frame_000042.jpg",
  "cited_rule": {
    "document":           "Factory Safety Manual v2.3",
    "section":            "4.2 Head Protection Requirements",
    "page":               17,
    "clause_summary":     "All personnel must wear a hard hat at all times on the production floor."
  },
  "risk_level":           "Critical",
  "risk_justification":   "Unprotected head in high falling-object risk zone.",
  "recommendation":       "Immediately issue and enforce mandatory helmet wear for all floor personnel.",
  "remediation_timeline": "Immediate",
  "narrative":            "A worker was detected at 00:01:23 on CAM-01 without a helmet in a mandatory hard-hat zone. This directly violates Section 4.2 of the Factory Safety Manual. The safety supervisor must intervene immediately and ensure the worker dons appropriate PPE before resuming work."
}
```

---

## 🧠 YOLOv8 Model Classes

Your `best.pt` model should be trained on (or fine-tuned for) these classes:

```
0: person
1: helmet
2: safety_vest
3: face_mask
4: gloves
5: fire_extinguisher
6: emergency_exit
7: restricted_zone_marker
```

---

## 📝 Notes

- The YOLO model (`models/best.pt`) is **not included** — add your own trained model.
- The knowledge base must be rebuilt whenever new PDFs are added to `data/raw/`.
- All violation screenshots are saved to `outputs/screenshots/` automatically.
- Reports can be exported as JSON from the Dashboard → Violation Reports tab.

## 🎯 Use Cases

- Construction Site Safety Monitoring
- Manufacturing Plant Compliance
- Industrial Safety Audits
- Smart Workplace Surveillance
- PPE Compliance Management

---

## Future Enhancements

- Live CCTV Integration
- SMS Alerts
- Mobile Application
- Multi-Camera Monitoring
- Advanced Analytics Dashboard
- Cloud Deployment

---

## 👨‍💻 Author

Sonu Kumar

LinkedIn: https://linkedin.com/in/sonukumar32977

---
