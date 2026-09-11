---
title: Agent 11 - Distributed Auction Task Allocation
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
---

# Agent 11 — Distributed Auction Task Allocation Agent

Distributed multi-agent task allocation using typed auctions, deterministic bidding rules, bounded reallocation, auditable coordination, and an optional bounded LLM execution layer.

## Status

**Deployment-ready.** The deterministic mission, centralized comparison baseline, stress evaluation harness, bounded LLM work handlers, and business-first Gradio demo are implemented and covered by automated tests. Hugging Face deployment is test-gated from GitHub `main` and safely skips until the deployment credential is configured.

## Problem

When several autonomous peers can perform the same task, a permanent semantic manager should not have to decide every assignment manually. This project implements a bounded task marketplace where peers independently decide whether to **BID** or **ABSTAIN**, application code validates eligibility, and a shared deterministic protocol selects the winner.

## Engineering Principle

> **Peers decide whether to compete. The protocol decides who wins.**

The model never becomes the auctioneer. Application code owns identity, capability registration, bid validation, scoring weights, tie-breaks, settlement, award lineage, reauction, escalation, evidence authority, and publication boundaries.

## Synthetic Business Scenario

The public demo uses a fictional corporate due-diligence mission for **Meridian Industrial Systems**. Six fictional peers have intentionally overlapping skills and compete for five auctionable tasks:

1. Market Attractiveness Research
2. Financial Quality Analysis
3. Operating & Execution Risk Review
4. Evidence Verification
5. Executive Due-Diligence Synthesis

The default deterministic mission allocates the tasks to **Atlas Research → Ledger Analyst → Sentinel Risk → Veritas Evidence → Quill Synthesis**.

## Deterministic Auction Policy

Valid bids are scored with one application-owned utility function:

```text
45% Capability + 25% Confidence + 20% Availability + 10% Cost Efficiency
```

Eligibility is evaluated before scoring. Deterministic tie-breaking uses:

1. higher total score,
2. higher registered capability,
3. lower estimated cost,
4. stable lexical `agent_id`.

Capability authority comes from the registry rather than bidder-provided claims.

## Controlled Architecture Comparison

The centralized and distributed implementations use the same peers, tasks, eligibility gates, scoring policy, executor, and bounded failure rules. Only allocation authority changes.

| Metric | Distributed Auction | Centralized Baseline |
|---|---:|---:|
| Mission success | Yes | Yes |
| Tasks completed | 5/5 | 5/5 |
| Execution cost | $347 | $347 |
| Allocation efficiency | 1.00 | 1.00 |
| Protocol/control messages | 50 | 10 |

In the clean full-information case, both architectures select the same workers. The distributed market pays additional communication overhead in exchange for decentralized allocation authority.

## Stress Evaluation

Nine deterministic scenarios test cost pressure, capability scarcity, low confidence, availability shock, exact ties, no-valid-bid escalation, single-failure recovery, and retry exhaustion.

Across the default stress suite:

- 7 scenarios succeed in both architectures,
- 2 scenarios correctly escalate in both architectures,
- worker disagreements: 0,
- aggregate synthetic execution cost: $2,639 in both architectures,
- mean allocation efficiency: 1.00,
- distributed protocol messages: 390,
- centralized control messages: 78.

## Bounded LLM Execution

LLM-assisted mode is optional and lives strictly inside the awarded-task execution boundary.

The model may draft only:

- title,
- summary,
- findings,
- approved evidence references.

The application still owns every allocation and publication decision. Model output is validated against a strict Pydantic schema, checked against an application-owned evidence allowlist, and then validated again through the typed `WorkProduct` contract. Invalid output fails closed into the existing bounded recovery path.

## Gradio Demo

**Live Space:** https://huggingface.co/spaces/FlyingNunchucks/11-distributed-auction-task-allocation-agent

The business-first demo exposes:

- **Task Marketplace** — task winners, scores, bid counts, rounds, messages, and execution cost,
- **Centralized vs Distributed** — controlled architecture tradeoff,
- **Stress Evaluation** — nine deterministic scenarios,
- **Work Products** — validated outputs from winning peers,
- **Protocol Audit** — append-only allocation events,
- **Engineering Boundary** — what the LLM may and may not control.

Deterministic mode is the safe default. LLM-assisted mode requires runtime configuration and fails closed if it is absent.

## Run Locally

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python app.py
```

## Hugging Face Deployment

GitHub is the source of truth. Pushes to `main` run the complete test suite first. Deployment runs only after tests pass and only when the GitHub repository secret `HF_DEPLOY_TOKEN` is configured.

Runtime LLM configuration belongs in the Hugging Face Space, not in source control:

- `HF_TOKEN` — Hugging Face Space secret used only for optional live inference.
- `MODEL_ID` — Space variable or secret identifying the inference model.
- `HF_BASE_URL` — optional non-secret override; defaults to the Hugging Face router endpoint.

Do not place secret values in `.env.example`, source files, commits, screenshots, logs, or README content.

## Repository Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── auction_coordination/
│       ├── auction.py
│       ├── baseline.py
│       ├── bidding.py
│       ├── demo.py
│       ├── evaluation.py
│       ├── execution.py
│       ├── hf_runtime.py
│       ├── llm.py
│       ├── models.py
│       ├── reauction.py
│       ├── registry.py
│       ├── runtime.py
│       ├── scenario.py
│       └── settlement.py
└── tests/
```

## Public-Safety Boundary

The repository and demo use synthetic, public-safe data only. No private curriculum material, client data, financial data, personal information, or real credentials belong in the public repository, fixtures, screenshots, logs, or Hugging Face Space.

## License

MIT
