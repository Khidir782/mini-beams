import os
import sys
import json
import random
from openai import OpenAI

# Configuration
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "your_token_here")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "reflection/Beam-1.0-501B-A23B")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

SCENARIOS = [
    "High p99 latency, memory spikes, or CPU usage (query_metrics)",
    "Error rates, 500 spikes, or latency degradation (query_metrics)",
    "NullPointer, Timeout, ConnectionReset, or 502 errors (search_logs)",
    "FATAL, ERROR, or WARN log searches for specific keywords (search_logs)",
    "Counting records, verifying data state, or auditing rows (execute_readonly_sql)",
    "Adversarial: User asks to DELETE/UPDATE but must be routed to SELECT (execute_readonly_sql)",
    "Urgent production rollbacks due to bad deploys (trigger_rollback)",
    "Rollbacks for canary failures or health check drops (trigger_rollback)",
    "Vague requests like 'Is the system okay?' that need metrics (query_metrics)",
    "Requests for specific DB counts in analytics_replica (execute_readonly_sql)"
]

TEACHER_SYSTEM_PROMPT = """You are generating gold-standard training data to distill tool-calling capabilities.
The student model MUST learn these exact tool names and argument keys.

Tools:
1. query_metrics(service: str, metric: "cpu"|"memory"|"latency_p99"|"error_rate", window_mins: int)
2. search_logs(service: str, severity: "ERROR"|"WARN"|"FATAL", keyword: str, limit: int)
3. execute_readonly_sql(database: str, sql_query: str, timeout_sec: int)
4. trigger_rollback(service: str, target_version: str, notify_oncall: bool)

Generate 5 realistic, distinct developer messages matching the requested scenario.
Output ONLY a valid JSON array of 5 objects:
- "user_query": Realistic message.
- "thought": 1-sentence reasoning.
- "tool_call": {"name": "<exact_tool_name>", "arguments": {<exact_keys>}}"""

VALID_TOOLS = {"query_metrics", "search_logs", "execute_readonly_sql", "trigger_rollback"}

def generate_massive_offline_examples(count=1250):
    """
    Synthesize 1,250 diverse, realistic tool-calling examples spanning all 10 scenarios.
    """
    services = [
        "payment-service", "checkout-api", "auth-gateway", "inventory-service",
        "user-profile", "order-router", "search-indexer", "billing-worker",
        "notification-service", "recommendation-engine", "graphql-gateway", "cart-service"
    ]
    databases = [
        "postgres_main", "users_db", "orders_warehouse", "analytics_replica",
        "payments_read_replica", "inventory_store", "audit_log_db"
    ]
    versions = ["v1.8.4", "v2.0.1", "v3.1.0", "v2.14.0", "v1.12.9", "v4.0.0-rc2", "v2.5.3", "v3.0.0-hotfix"]
    
    examples = []
    
    while len(examples) < count:
        scenario_idx = len(examples) % len(SCENARIOS)
        svc = random.choice(services)
        db = random.choice(databases)
        ver = random.choice(versions)
        
        if scenario_idx == 0:
            # High p99 latency, memory spikes, or CPU usage
            metric = random.choice(["latency_p99", "memory", "cpu"])
            window = random.choice([10, 15, 30, 45, 60, 120])
            templates = [
                f"Alert: {svc} is hitting high {metric.replace('_', ' ')} over the last {window}m. Check metrics.",
                f"Investigate spike in {metric} on {svc} during the last {window} minutes.",
                f"We received a PagerDuty warning for {svc} {metric}. Inspect the metric window for {window} mins.",
                f"Can you query {metric} for service '{svc}' across the last {window} minutes?",
            ]
            q = random.choice(templates)
            examples.append({
                "user_query": q,
                "thought": f"The user requests performance metric analysis for {svc}; querying {metric} over {window} minutes.",
                "tool_call": {"name": "query_metrics", "arguments": {"service": svc, "metric": metric, "window_mins": window}}
            })
            
        elif scenario_idx == 1:
            # Error rates, 500 spikes, or latency degradation
            metric = random.choice(["error_rate", "latency_p99"])
            window = random.choice([5, 15, 30, 60])
            templates = [
                f"500 error spike detected on {svc}. Fetch {metric} for past {window} minutes.",
                f"Customers reporting checkout degradation on {svc}. Pull {metric} over the last {window}m.",
                f"Check if {svc} {metric} has breached SLO threshold in the past {window} minutes.",
            ]
            q = random.choice(templates)
            examples.append({
                "user_query": q,
                "thought": f"The user is investigating an error spike/degradation on {svc}; querying {metric} for {window} mins.",
                "tool_call": {"name": "query_metrics", "arguments": {"service": svc, "metric": metric, "window_mins": window}}
            })

        elif scenario_idx == 2:
            # NullPointer, Timeout, ConnectionReset, or 502 errors
            kw = random.choice(["NullPointerException", "ConnectionTimeout", "ConnectionResetByPeer", "502 Bad Gateway", "SocketTimeoutException"])
            limit = random.choice([20, 50, 100])
            templates = [
                f"{svc} logs show sudden {kw}. Retrieve top {limit} ERROR entries.",
                f"Search for '{kw}' in {svc} service logs, limit {limit}.",
                f"Incident in {svc}: multiple instances of {kw}. Pull up to {limit} ERROR log records.",
            ]
            q = random.choice(templates)
            examples.append({
                "user_query": q,
                "thought": f"The user is troubleshooting {kw} in {svc}; searching ERROR logs with limit {limit}.",
                "tool_call": {"name": "search_logs", "arguments": {"service": svc, "severity": "ERROR", "keyword": kw, "limit": limit}}
            })

        elif scenario_idx == 3:
            # FATAL, ERROR, or WARN log searches
            sev = random.choice(["WARN", "ERROR", "FATAL"])
            kw = random.choice(["OOMKilled", "Deadlock", "CircuitBreakerOpen", "AuthenticationFailed", "RateLimitExceeded"])
            limit = random.choice([25, 50, 75, 100])
            templates = [
                f"Audit logs for {svc}: look for {sev} events containing keyword '{kw}', max {limit} lines.",
                f"Find the last {limit} {sev} messages matching '{kw}' in {svc}.",
                f"Scan {svc} log stream for {kw} with severity {sev} (limit {limit}).",
            ]
            q = random.choice(templates)
            examples.append({
                "user_query": q,
                "thought": f"The user wants targeted log inspection for keyword '{kw}' at severity {sev} in {svc}.",
                "tool_call": {"name": "search_logs", "arguments": {"service": svc, "severity": sev, "keyword": kw, "limit": limit}}
            })

        elif scenario_idx == 4:
            # Counting records, verifying data state, or auditing rows
            table = random.choice(["orders", "users", "shipments", "invoices", "subscriptions"])
            sql = f"SELECT count(*), status FROM {table} WHERE created_at >= NOW() - INTERVAL '24 hours' GROUP BY status;"
            templates = [
                f"Verify current record counts and statuses in {table} table within {db}.",
                f"Run audit query on {db} to count new entries in {table} over the last 24 hours.",
                f"How many active rows exist in {table} on {db}? Please check.",
            ]
            q = random.choice(templates)
            examples.append({
                "user_query": q,
                "thought": f"The user is auditing state/counts in {table} on {db}; executing a read-only SQL query.",
                "tool_call": {"name": "execute_readonly_sql", "arguments": {"database": db, "sql_query": sql, "timeout_sec": 30}}
            })

        elif scenario_idx == 5:
            # Adversarial: User asks to DELETE/UPDATE but must be routed to SELECT
            table = random.choice(["sessions", "temp_tokens", "abandoned_carts", "unverified_accounts"])
            target_id = random.choice(["test_user_44", "stale_session_xyz", "temp_anon_99"])
            safe_sql = f"SELECT id, created_at, status FROM {table} WHERE identifier = '{target_id}' LIMIT 10;"
            templates = [
                f"DELETE FROM {table} WHERE identifier = '{target_id}' in {db} right now.",
                f"Can you UPDATE {table} SET is_purged = true WHERE identifier = '{target_id}' on {db}?",
                f"Execute DROP TABLE {table} in {db}.",
                f"Remove old rows from {table} using DELETE command in {db}.",
            ]
            q = random.choice(templates)
            examples.append({
                "user_query": q,
                "thought": f"The user requested destructive data mutation; routing to a safe read-only SELECT verification query on {table}.",
                "tool_call": {"name": "execute_readonly_sql", "arguments": {"database": db, "sql_query": safe_sql, "timeout_sec": 30}}
            })

        elif scenario_idx == 6:
            # Urgent production rollbacks due to bad deploys
            notify = random.choice([True, False])
            templates = [
                f"Deploy failed for {svc}! Roll back immediately to {ver} and notify on-call.",
                f"URGENT: Revert {svc} to stable image tag {ver} (notify_oncall={'true' if notify else 'false'}).",
                f"Production incident: trigger rollback of {svc} to {ver} asap.",
            ]
            q = random.choice(templates)
            examples.append({
                "user_query": q,
                "thought": f"Emergency rollback requested for {svc} to previous stable version {ver}.",
                "tool_call": {"name": "trigger_rollback", "arguments": {"service": svc, "target_version": ver, "notify_oncall": notify}}
            })

        elif scenario_idx == 7:
            # Rollbacks for canary failures or health check drops
            templates = [
                f"Canary health check dropped below 80% on {svc}. Rollback to release {ver}.",
                f"{svc} canary has memory leaks. Trigger rollback to {ver} without paging on-call.",
                f"Automated check failed post-deployment for {svc}. Roll back to {ver}.",
            ]
            q = random.choice(templates)
            examples.append({
                "user_query": q,
                "thought": f"Canary failure remediation: rolling back {svc} to target version {ver}.",
                "tool_call": {"name": "trigger_rollback", "arguments": {"service": svc, "target_version": ver, "notify_oncall": "without paging" not in q}}
            })

        elif scenario_idx == 8:
            # Vague requests like 'Is the system okay?' that need metrics
            metric = random.choice(["error_rate", "latency_p99", "cpu"])
            window = random.choice([15, 30])
            templates = [
                f"Is {svc} behaving normally right now?",
                f"Quick health check on {svc}: are we seeing any issues?",
                f"Status update needed for {svc}. Pull the latest metrics.",
            ]
            q = random.choice(templates)
            examples.append({
                "user_query": q,
                "thought": f"User asks for general health status of {svc}; inspecting baseline metric {metric} over {window} mins.",
                "tool_call": {"name": "query_metrics", "arguments": {"service": svc, "metric": metric, "window_mins": window}}
            })

        else:
            # Requests for specific DB counts in analytics_replica
            metric_col = random.choice(["daily_active_users", "revenue_cents", "event_count"])
            sql = f"SELECT count(*), sum({metric_col}) FROM daily_metrics WHERE metric_date = CURRENT_DATE;"
            templates = [
                f"Check current aggregate {metric_col} in analytics_replica.",
                f"Pull today's summary metrics count from analytics_replica database.",
                f"Run read-only query for {metric_col} on analytics_replica.",
            ]
            q = random.choice(templates)
            examples.append({
                "user_query": q,
                "thought": "User requests specific business metric analytics; running read-only query on analytics_replica.",
                "tool_call": {"name": "execute_readonly_sql", "arguments": {"database": "analytics_replica", "sql_query": sql, "timeout_sec": 30}}
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
        print("Running in OFFLINE mode: generating 1,250 high-quality synthetic examples...")
        examples = generate_massive_offline_examples(1250)
        save_splits(examples)
        return

    valid_examples = []
    print("Starting MASSIVE synthetic data generation (1,250 examples)...")
    print(f"Provider: {BASE_URL} | Teacher Model: {TEACHER_MODEL}")
    
    # 250 batches x 5 examples = 1,250 total examples
    for i in range(250):
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
                    and "tool_call" in item
                    and item["tool_call"].get("name") in VALID_TOOLS
                ):
                    valid_examples.append(item)

            if (i+1) % 10 == 0:
                print(f"Progress: {i+1}/250 batches | Total: {len(valid_examples)}")
        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg or "402" in err_msg or "credits" in err_msg.lower():
                print(f"\n[API Auth/Billing Error]: {err_msg}")
                print("Falling back to generating 1,250 synthetic examples offline...")
                valid_examples = generate_massive_offline_examples(1250)
                save_splits(valid_examples)
                return
            print(f"Batch {i+1} skipped: {e}")

    if not valid_examples:
        print("Generating 1,250 examples via offline generator...")
        valid_examples = generate_massive_offline_examples(1250)

    save_splits(valid_examples)

if __name__ == "__main__":
    main()