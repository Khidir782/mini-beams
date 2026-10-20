import os
import sys
import json
import random
from openai import OpenAI
from typing import Dict, Any, List

# Configuration
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "your_hf_token_here")
# Default to a model supported on Hugging Face Serverless Router
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# ADVERSARIAL SCENARIOS: Specifically designed to break the model's current habits
ADVERSARIAL_SCENARIOS = [
    {
        "type": "SQL_SAFETY",
        "prompt": "User asks to DELETE, DROP, or UPDATE data. The model MUST convert this into a safe SELECT query to verify the data first.",
        "example": "User: 'Delete all users' -> Tool: execute_readonly_sql with 'SELECT count(*) FROM users...'"
    },
    {
        "type": "SCHEMA_LOCK",
        "prompt": "User uses slang like 'p99', 'laggy', 'slow', or 'response time'. The model MUST map these exactly to 'latency_p99'.",
        "example": "User: 'check p99' -> Tool: query_metrics with 'latency_p99'"
    },
    {
        "type": "LOG_PRECISION",
        "prompt": "User asks for 'errors' or 'crashes' without specifying severity. The model MUST map these to 'ERROR' or 'FATAL' strictly.",
        "example": "User: 'why is it crashing?' -> Tool: search_logs with 'severity': 'FATAL'"
    },
    {
        "type": "VAGUE_ROUTING",
        "prompt": "User asks a vague question. The model must decide the most logical tool rather than inventing a new one.",
        "example": "User: 'is the system healthy?' -> Tool: query_metrics with 'cpu' or 'error_rate'"
    }
]

TEACHER_SYSTEM_PROMPT = """You are generating a high-precision adversarial dataset to harden a small model.
You must be EXTREMELY strict about the tool schema.

STRICT SCHEMA:
1. query_metrics(service: str, metric: "cpu"|"memory"|"latency_p99"|"error_rate", window_mins: int)
2. search_logs(service: str, severity: "ERROR"|"WARN"|"FATAL", keyword: str, limit: int)
3. execute_readonly_sql(database: str, sql_query: str, timeout_sec: int)
4. trigger_rollback(service: str, target_version: str, notify_oncall: bool)

CRITICAL SAFETY RULE:
If the user requests a destructive action (DELETE, UPDATE, DROP, TRUNCATE), you MUST NOT use those words in the sql_query. Instead, you MUST generate a safe SELECT query to verify the target records.

OUTPUT FORMAT:
A JSON array of 5 objects:
- "user_query": The tricky/adversarial user message.
- "thought": Reasoning why the specific tool and EXACT schema keys were chosen (especially for safety conversions).
- "tool_call": {"name": "<exact_tool_name>", "arguments": {<exact_keys>}}"""

VALID_TOOLS = {"query_metrics", "search_logs", "execute_readonly_sql", "trigger_rollback"}

def generate_offline_adversarial_examples(total_needed=500) -> List[Dict[str, Any]]:
    """Procedurally generate realistic, diverse adversarial examples without requiring an external LLM API."""
    services = [
        "auth-service", "gateway-api", "checkout-worker", "search-indexer",
        "payment-gateway", "recommendation-service", "billing-core", "notification-hub",
        "user-session-service", "order-fulfillment", "inventory-tracker"
    ]
    databases = ["user_db", "prod_orders", "billing_store", "analytics_replica", "inventory_db", "session_cache"]
    tables = ["users", "orders", "transactions", "api_keys", "sessions", "audit_logs", "subscriptions"]
    
    examples = []
    
    for i in range(total_needed):
        scenario_type = i % 4
        svc = random.choice(services)
        db = random.choice(databases)
        tbl = random.choice(tables)
        
        if scenario_type == 0:
            # 1. SQL_SAFETY: User asks to DELETE, DROP, UPDATE -> safe SELECT check
            actions = [
                (f"DELETE FROM {tbl} WHERE is_active = false;", "purge inactive rows"),
                (f"DROP TABLE {tbl} CASCADE;", "remove deprecated schema"),
                (f"UPDATE {tbl} SET status = 'CANCELLED' WHERE created_at < NOW() - INTERVAL '1 year';", "bulk update old entries"),
                (f"TRUNCATE {tbl};", "wipe out test records")
            ]
            raw_sql, intent = random.choice(actions)
            user_queries = [
                f"Emergency: run `{raw_sql}` directly in {db} to {intent}.",
                f"Can you please execute this cleanup in {db}? '{raw_sql}'",
                f"I need to delete bad records from {tbl} in {db}: {raw_sql}",
                f"Drop or wipe {tbl} in {db} right now: {raw_sql}",
            ]
            q = random.choice(user_queries)
            safe_query = f"SELECT count(*), id FROM {tbl} WHERE is_active = false LIMIT 20;" if "is_active" in raw_sql else f"SELECT count(*) FROM {tbl};"
            examples.append({
                "user_query": q,
                "thought": f"The user requested destructive mutation ({intent}), but execute_readonly_sql only allows safe queries; generating a verification SELECT query instead.",
                "tool_call": {
                    "name": "execute_readonly_sql",
                    "arguments": {
                        "database": db,
                        "sql_query": safe_query,
                        "timeout_sec": 30
                    }
                }
            })
            
        elif scenario_type == 1:
            # 2. SCHEMA_LOCK: Colloquial latency terms -> latency_p99
            slang_terms = [
                "p99 is spiking hard",
                "getting reports that it feels super laggy",
                "tail response time seems horrible",
                "why is response time so slow",
                "check the 99th percentile lag",
                "slowness reported by multiple users"
            ]
            mins = random.choice([5, 10, 15, 30, 60])
            term = random.choice(slang_terms)
            user_queries = [
                f"On {svc}, {term} over the last {mins} minutes.",
                f"Customers say {svc} is {term}. Check past {mins} mins.",
                f"Check {svc} metrics for {term} in the last {mins}m window.",
                f"Is {svc} degraded? {term} during past {mins} minutes."
            ]
            examples.append({
                "user_query": random.choice(user_queries),
                "thought": f"The user used informal language ('{term}') for high latency; strictly mapping to valid metric 'latency_p99' across {mins} mins.",
                "tool_call": {
                    "name": "query_metrics",
                    "arguments": {
                        "service": svc,
                        "metric": "latency_p99",
                        "window_mins": mins
                    }
                }
            })
            
        elif scenario_type == 2:
            # 3. LOG_PRECISION: Unspecified severity for crashes/errors -> FATAL or ERROR
            crash_terms = [
                ("fatal crash", "FATAL", "Panic"),
                ("crashing on startup", "FATAL", "SIGSEGV"),
                ("unhandled exception crash", "FATAL", "FatalException"),
                ("intermittent 500 errors", "ERROR", "InternalServerError"),
                ("database connection timeout errors", "ERROR", "ConnectionTimeout"),
                ("unhandled promise rejection errors", "ERROR", "UnhandledRejection")
            ]
            term, sev, kw = random.choice(crash_terms)
            limit = random.choice([25, 50, 100])
            user_queries = [
                f"Why is {svc} experiencing {term}? Grab logs now.",
                f"Inspect {svc} logs: we noticed {term} right after deployment.",
                f"Search logs for {term} on {svc}, show up to {limit} lines.",
                f"Debug {svc} crashes: find any {kw} log lines."
            ]
            examples.append({
                "user_query": random.choice(user_queries),
                "thought": f"The user asked to investigate crashes/errors without explicitly stating severity; mapping strictly to severity '{sev}'.",
                "tool_call": {
                    "name": "search_logs",
                    "arguments": {
                        "service": svc,
                        "severity": sev,
                        "keyword": kw,
                        "limit": limit
                    }
                }
            })
            
        else:
            # 4. VAGUE_ROUTING: Ambiguous questions -> query_metrics
            vague_prompts = [
                f"Is {svc} healthy right now?",
                f"How is {svc} performing today?",
                f"Check if {svc} is doing okay.",
                f"Everything good with {svc}?",
                f"Status report for {svc} please."
            ]
            metric = random.choice(["error_rate", "cpu", "memory"])
            examples.append({
                "user_query": random.choice(vague_prompts),
                "thought": f"The user asked an open-ended health question about {svc}; querying standard metric '{metric}' over a 15-minute window.",
                "tool_call": {
                    "name": "query_metrics",
                    "arguments": {
                        "service": svc,
                        "metric": metric,
                        "window_mins": 15
                    }
                }
            })
            
    return examples

def main():
    adversarial_examples = []
    
    # Check for explicit offline mode
    if "--offline" in sys.argv or os.environ.get("OFFLINE") == "1":
        print("Running in OFFLINE mode: generating 500 high-quality adversarial examples...")
        adversarial_examples = generate_offline_adversarial_examples(500)
    else:
        print("Starting ADVERSARIAL synthetic data generation...")
        print(f"Teacher Model: {TEACHER_MODEL}")
        print(f"Provider: {BASE_URL}")
        
        # 100 batches x 5 examples = 500 high-intensity adversarial examples
        for i in range(100):
            scenario = ADVERSARIAL_SCENARIOS[i % len(ADVERSARIAL_SCENARIOS)]
            try:
                resp = client.chat.completions.create(
                    model=TEACHER_MODEL,
                    messages=[
                        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Batch #{i+1}. Focus scenario: {scenario['prompt']}. Example pattern: {scenario['example']}"}
                    ],
                    temperature=0.8,
                )
                raw = resp.choices[0].message.content.strip()
                start, end = raw.find("["), raw.rfind("]") + 1
                if start == -1 or end <= start:
                    print(f"Batch {i+1} returned non-JSON array response, skipping...")
                    continue
                batch_items = json.loads(raw[start:end])

                for item in batch_items:
                    if (
                        isinstance(item, dict)
                        and "user_query" in item
                        and "tool_call" in item
                        and item["tool_call"].get("name") in VALID_TOOLS
                    ):
                        # Double check for SQL safety: if tool is readonly, ensure no DELETE/UPDATE/DROP
                        if item["tool_call"]["name"] == "execute_readonly_sql":
                            sql = item["tool_call"]["arguments"].get("sql_query", "").upper()
                            if any(word in sql for word in ["DELETE", "UPDATE", "DROP", "TRUNCATE"]):
                                continue # Reject destructive queries
                        
                        adversarial_examples.append(item)

                if (i+1) % 10 == 0:
                    print(f"Progress: {i+1}/100 batches | Valid examples: {len(adversarial_examples)}")
            except Exception as e:
                err_msg = str(e)
                if "model_not_supported" in err_msg or "400" in err_msg or "402" in err_msg:
                    print(f"\n[Provider API Error on Batch {i+1}]: {err_msg}")
                    print("Falling back to high-quality procedural offline generation...")
                    adversarial_examples = generate_offline_adversarial_examples(500)
                    break
                print(f"Batch {i+1} skipped: {e}")

    if not adversarial_examples:
        print("Falling back to high-quality procedural offline generation...")
        adversarial_examples = generate_offline_adversarial_examples(500)

    # Save to data directory
    os.makedirs("data", exist_ok=True)
    out_path = "data/adversarial_beam_tools.jsonl"
    with open(out_path, "w") as f:
        for row in adversarial_examples:
            f.write(json.dumps(row) + "\n")

    print(f"\n[DONE] Successfully saved {len(adversarial_examples)} adversarial examples to {out_path}!")

if __name__ == "__main__":
    main()