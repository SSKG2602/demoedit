# Safety & Scope (Lite)

**In-scope**

- Public PoC for time-aware retrieval
- Deterministic fallbacks when no LLM is available
- Transparent evidence exposure via Attribution Cards

**Out-of-scope**

- Proprietary controllers/agents and their training data
- Full GraphRAG claim resolution, conflict arbitration
- Private or sensitive datasets
- Real-time web scraping or external API calls

**Data handling**

- Demo corpus is local files only.
- No PII expected; if present in user data, responsibility lies with the user.
- No background data collection.

**Model behavior**

- The LLM is local via HF. If it fails, the system returns a compact, evidence-based fallback.
- No harmful content generation features included.

**Operational**

- Designed to run offline.
- Observability prints timing and counters; OTEL is optional.
