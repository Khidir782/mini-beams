# Mini-Beams: Knowledge Distillation for Tool-Calling Agents

This repository implements a knowledge distillation pipeline designed to compress the reasoning and tool-calling capabilities of a massive-scale model (`Beam-1.0-501B-A23B`) into a highly efficient, sub-1B parameter student (`Qwen3.5-0.8B`).

The objective was to create a specialized DevOps & Database Incident Triage agent capable of strict schema adherence and safety-aware routing, deployable on edge hardware.

## Core Objective
The primary challenge was to maintain the "reasoning DNA" of the 501B teacher while eliminating the "reasoning noise" (monologues and simulated responses) in the student. The final goal was a model that behaves as a pure mathematical function: `User Query` $\rightarrow$ `Strict JSON Tool Call`.

## Architecture & Pipeline

### 1. Knowledge Distillation (SFT)
We utilized Supervised Fine-Tuning (SFT) to transfer capabilities. The process involved:
- **Teacher Synthesis**: Generating 1,500+ high-signal reasoning traces from the 501B Beam model.
- **SFT Optimization**: Using **Unsloth** for memory-efficient training, applying LoRA (Rank-16) across all linear targets.
- **Formatting Lock**: Implementing a strict prompt-response pair to force the model to adhere to a specific JSON schema, preventing "Schema Collapse."

### 2. Adversarial Hardening
To move the model from "prototype" to "production-ready," we implemented adversarial training to address two critical failure modes:
- **Semantic Leakage**: Training the model to resist "destructive" requests. For example, converting `DELETE` requests into safe `SELECT` verification queries.
- **Schema Drift**: Mapping colloquialisms (e.g., "laggy," "p99") to strict API keys (`latency_p99`).

### 3. Quantization & Deployment
To enable local execution on consumer hardware, the model was quantized:
- **Format**: GGUF (via llama.cpp).
- **Method**: 4-bit NF4 quantization (`q4_k_m`).
- **Footprint**: Reduced from ~1.6GB to ~500MB.

## Results

| Metric | Baseline (Zero-Shot) | Distilled (General) | Hardened (Adversarial) |
| :--- | :---: | :---: | :---: |
| **Valid JSON %** | Low | 92.0% | **100.0%** |
| **Correct Tool %** | Low | 92.0% | **100.0%** |
| **Exact Schema Match %** | Low | 92.0% | **100.0%** |

## Security Analysis & Vulnerability Assessment

During the hardening phase, a critical vulnerability was identified regarding **Semantic Leakage**. Despite the model being trained to use a `readonly_sql` tool, adversarial prompts revealed that the model could still generate destructive SQL statements (e.g., `DELETE`, `DROP`) within the arguments of a read-only tool.

### The Vulnerability: "Confused Deputy"
The model acts as a "Confused Deputy"—it has the permission to call the `execute_readonly_sql` tool, but it uses that permission to pass a destructive payload to the backend. This demonstrates that **model obedience is not a security guarantee.**

### Mitigation Strategy
To mitigate this in a production environment, a **Deterministic Validation Layer** must be implemented between the LLM and the execution engine:
1. **AST Parsing**: Using a SQL parser (e.g., `sqlglot`) to analyze the generated query.
2. **Keyword Blacklisting**: Blocking any query containing `DELETE`, `UPDATE`, `DROP`, or `TRUNCATE`.
3. **Least Privilege**: Ensuring the database user associated with the tool has physically restricted permissions (Read-Only) at the database level.

## Usage

### Local Deployment via Ollama
The model is available on the Hugging Face Hub. You can run it locally using Ollama:

```bash
ollama run hf.co/Khide/Qwen3.5-0.8B-Beam-ToolRouter-GGUF
