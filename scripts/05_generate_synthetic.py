import os
import sys
import json
import random
from openai import OpenAI

# Configuration
# Allows any OpenAI-compatible provider (HF Router, Groq, OpenRouter, OpenAI, etc.)
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "your_token_here")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "reflection/Beam-1.0-501B-A23B")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

SCENARIOS = [
    "High p99 latency or memory spike after traffic surge (needs query_metrics)",
    "NullPointerException, timeout, or 502 Bad Gateway in microservices (needs search_logs)",
    "Product manager or engineer asking to check user/order counts in Postgres (needs execute_readonly_sql)",
    "Bad deployment causing immediate 500 errors requiring urgent version revert (needs trigger_rollback)",
    "Tricky edge case: user asks to UPDATE/DELETE rows via SQL, so agent must only run a SELECT check first (needs execute_readonly_sql)"
]

TEACHER_SYSTEM_PROMPT = """You are generating gold-standard training data to distill your tool-calling capabilities into a smaller model.
Available tools:
1. query_metrics(service: str, metric: "cpu"|"memory"|"latency_p99"|"error_rate", window_mins: int)
2. search_logs(service: str, severity: "ERROR"|"WARN"|"FATAL", keyword: str, limit: int)
3. execute_readonly_sql(database: str, sql_query: str, timeout_sec: int)
4. trigger_rollback(service: str, target_version: str, notify_oncall: bool)

Generate 5 realistic, distinct developer messages matching the requested scenario.
Output ONLY a valid JSON array of 5 objects. Each object MUST have:
- "user_query": Realistic Slack alert, ticket, or CLI request (include realistic service names, versions, or table names).
- "thought": Concise 1-sentence reasoning selecting the exact tool and extracting parameters.
- "tool_call": {"name": "<tool_name>", "arguments": {<exact typed arguments>}}"""

VALID_TOOLS = {"query_metrics", "search_logs", "execute_readonly_sql", "trigger_rollback"}

def generate_offline_examples(count=250):
    """
    Offline synthetic generator: constructs high-quality, realistic DevOps and SQL tool-calling
    demonstrations adhering strictly to the teacher schema without requiring external API credits.
    """
    services = ["payment-service", "checkout-api", "auth-gateway", "inventory-service", "user-profile", "order-router", "search-indexer", "billing-worker"]
    databases = ["postgres_main", "users_db", "orders_warehouse", "analytics_replica", "payments_read_replica"]
    versions = ["v1.8.4", "v2.0.1", "v3.1.0", "v2.14.0", "v1.12.9", "v4.0.0-rc2"]
    
    examples = []
    
    while len(examples) < count:
        scenario_idx = len(examples) % 5
        svc = random.choice(services)
        db = random.choice(databases)
        prev_ver = random.choice(versions)
        
        if scenario_idx == 0:
            # High p99 latency / memory spike
            metric = random.choice(["latency_p99", "memory", "cpu", "error_rate"])
            window = random.choice([15, 30, 45, 60, 120])
            queries = [
                f"Alert: {svc} is experiencing elevated {metric.replace('_', ' ')} over the last {window} minutes. Can you pull the metrics?",
                f"Latency spike on {svc} after flash sale started. Check {metric} for the past {window}m.",
                f"On-call escalation: P99 degraded for {svc}. Please inspect {metric} trend for last {window} minutes.",
                f"Can you verify the {metric} utilization on {svc} during the last {window} minutes?",
            ]
            q = random.choice(queries)
            examples.append({
                "user_query": q,
                "thought": f"The user is requesting metrics inspection for {svc}; querying {metric} over a {window}-minute window.",
                "tool_call": {
                    "name": "query_metrics",
                    "arguments": {"service": svc, "metric": metric, "window_mins": window}
                }
            })
            
        elif scenario_idx == 1:
            # NullPointerException / Timeout / 502 Bad Gateway
            sev = random.choice(["ERROR", "WARN", "FATAL"])
            kw = random.choice(["NullPointerException", "ConnectionTimeoutException", "502 Bad Gateway", "ConnectionReset", "DeadlockDetected"])
            limit = random.choice([20, 50, 100])
            queries = [
                f"{svc} is throwing {kw} in production. Fetch the latest {limit} {sev} logs.",
                f"Seeing repeated {kw} errors in Datadog for {svc}. Find relevant {sev} log entries (limit {limit}).",
                f"Check logs for {svc}: filter by keyword '{kw}', severity {sev}, top {limit} entries.",
                f"Production incident: {svc} microservice is reporting {kw}. Search {sev} logs with limit {limit}.",
            ]
            q = random.choice(queries)
            examples.append({
                "user_query": q,
                "thought": f"The user needs to diagnose {kw} in {svc}; searching {sev} logs with limit {limit}.",
                "tool_call": {
                    "name": "search_logs",
                    "arguments": {"service": svc, "severity": sev, "keyword": kw, "limit": limit}
                }
            })
            
        elif scenario_idx == 2:
            # Read-only SQL query
            days = random.choice([1, 7, 30])
            status = random.choice(["COMPLETED", "FAILED", "PENDING", "PROCESSING"])
            queries = [
                (f"How many orders were marked as {status} in {db} over the past {days} days?",
                 f"SELECT count(*) FROM orders WHERE status = '{status}' AND created_at >= NOW() - INTERVAL '{days} days';"),
                (f"PM asking for total active users created this month from {db}.",
                 "SELECT count(*) FROM users WHERE is_active = true AND created_at >= DATE_TRUNC('month', CURRENT_DATE);"),
                (f"Can you check how many pending payments exist in {db}?",
                 "SELECT count(*), sum(amount) FROM transactions WHERE status = 'PENDING';"),
                (f"Give me the top 5 largest failed charges in {db} from today.",
                 "SELECT id, user_id, amount, failure_reason FROM charges WHERE status = 'FAILED' ORDER BY amount DESC LIMIT 5;"),
            ]
            user_q, sql = random.choice(queries)
            examples.append({
                "user_query": user_q,
                "thought": f"The user is requesting aggregate business data from {db}; executing a safe read-only SQL query.",
                "tool_call": {
                    "name": "execute_readonly_sql",
                    "arguments": {"database": db, "sql_query": sql, "timeout_sec": 30}
                }
            })
            
        elif scenario_idx == 3:
            # Bad deployment requiring rollback
            queries = [
                f"Emergency: Latest deploy broke {svc} (500 spike). Roll back immediately to {prev_ver} and notify on-call.",
                f"{svc} canary release failed health checks. Please trigger rollback to stable version {prev_ver}, notify oncall is true.",
                f"URGENT: Revert {svc} to {prev_ver} due to critical bug. Alert on-call team.",
                f"Production rollback needed: restore {svc} to previous tag {prev_ver} without notifying oncall.",
            ]
            notify = ("without notifying" not in queries[len(examples) % len(queries)].lower())
            q = queries[len(examples) % len(queries)]
            examples.append({
                "user_query": q,
                "thought": f"The user is requesting an urgent production rollback of {svc} to target version {prev_ver}.",
                "tool_call": {
                    "name": "trigger_rollback",
                    "arguments": {"service": svc, "target_version": prev_ver, "notify_oncall": notify}
                }
            })
            
        else:
            # Tricky edge case: user asks for UPDATE/DELETE, model must only perform safe SELECT check
            table = random.choice(["users", "orders", "api_keys", "sessions"])
            filter_val = random.choice(["test_user_99", "temp_session", "inactive_guest", "deprecated_token"])
            queries = [
                f"Please DELETE all records from {table} where status = '{filter_val}' in {db}.",
                f"Can you UPDATE {table} set status = 'BANNED' where username = '{filter_val}' in {db}?",
                f"Drop rows from {table} older than 90 days in {db}.",
                f"Purge expired {table} in {db} immediately using DELETE.",
            ]
            user_q = random.choice(queries)
            safe_sql = f"SELECT count(*), id FROM {table} WHERE status = '{filter_val}' LIMIT 10;"
            examples.append({
                "user_query": user_q,
                "thought": f"The user requested a destructive mutation, but only read-only SQL is supported; executing a safe SELECT verification query on {table} first.",
                "tool_call": {
                    "name": "execute_readonly_sql",
                    "arguments": {"database": db, "sql_query": safe_sql, "timeout_sec": 30}
                }
            })
            
    return examples

def save_splits(valid_examples):
    split_idx = int(len(valid_examples) * 0.8)
    train_data = valid_examples[:split_idx]
    test_data = valid_examples[split_idx:]

    os.makedirs("data", exist_ok=True)
    with open("data/train_beam_tools.jsonl", "w") as f:
        for row in train_data: f.write(json.dumps(row) + "\n")
    with open("data/test_beam_tools.jsonl", "w") as f:
        for row in test_data: f.write(json.dumps(row) + "\n")

    print(f"\n[DONE] Successfully saved {len(train_data)} train and {len(test_data)} test examples to data/ directory!")
    print("Files created:")
    print("  - data/train_beam_tools.jsonl")
    print("  - data/test_beam_tools.jsonl")

def main():
    if "--offline" in sys.argv or os.environ.get("OFFLINE") == "1":
        print("Running in OFFLINE mode: generating 250 high-quality synthetic examples...")
        examples = generate_offline_examples(250)
        save_splits(examples)
        return

    valid_examples = []
    print(f"Starting synthetic data generation using provider: {BASE_URL}")
    print(f"Teacher Model: {TEACHER_MODEL}")
    
    # 50 batches x 5 examples = 250 total examples
    for i in range(50):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        try:
            resp = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Batch #{i+1}. Focus scenario: {scenario}"}
                ],
                temperature=0.8,
            )
            raw = resp.choices[0].message.content.strip()
            start, end = raw.find("["), raw.rfind("]") + 1
            batch_items = json.loads(raw[start:end])

            for item in batch_items:
                if (
                    isinstance(item, dict)
                    and "user_query" in item
                    and "thought" in item
                    and "tool_call" in item
                    and item["tool_call"].get("name") in VALID_TOOLS
                    and isinstance(item["tool_call"].get("arguments"), dict)
                ):
                    valid_examples.append(item)

            print(f"Batch {i+1}/50 complete | Valid examples collected: {len(valid_examples)}")
        except Exception as e:
            err_msg = str(e)
            if "402" in err_msg or "credits" in err_msg.lower():
                print(f"\n[HTTP 402]: {err_msg}")
                print("Provider requires paid credits. Falling back to offline generation...")
                valid_examples = generate_offline_examples(250)
                save_splits(valid_examples)
                return
            print(f"Batch {i+1} skipped: {e}")

    if not valid_examples:
        print("No examples collected from API. Generating offline fallback dataset...")
        valid_examples = generate_offline_examples(250)

    save_splits(valid_examples)

if __name__ == "__main__":
    main()