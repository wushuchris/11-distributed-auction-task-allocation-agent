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

**Complete and production-validated.** The deterministic mission, centralized comparison baseline, stress evaluation harness, bounded LLM work handlers, business-first Gradio demo, test-gated deployment path, and live Hugging Face runtime have all been validated. The final automated suite passes **140 tests**, and both deterministic and live LLM-assisted production checks completed successfully.

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

The default mission allocates the tasks to **Atlas Research → Ledger Analyst → Sentinel Risk → Veritas Evidence → Quill Synthesis**.

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

> **Validation determines who may compete. Scoring determines which valid competitor wins.**

## Bounded Reauction

Explicit execution failure can reopen the same logical auction for one additional round. The failed worker is excluded from the immediate retry by protocol admission rather than by mutating its long-term registry profile.

The bounded lifecycle is:

```text
ANNOUNCED → BIDDING → AWARDED → IN_PROGRESS → COMPLETED
                                  ↓
                                FAILED
                                  ↓
                            REAUCTIONING
                                  ↓
                          BIDDING (round + 1)
                                  ↓
                         COMPLETED / ESCALATED
```

This is intentionally narrow recovery behavior, not a claim of general fault tolerance.

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

> **The centralized allocator spends authority. The distributed auction spends messages.**

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

Under equivalent information and utility policy, the distributed auction preserves allocation quality and failure behavior in these scenarios while paying a measurable communication premium for decentralized allocation authority.

## Bounded LLM Execution

LLM-assisted mode lives strictly inside the awarded-task execution boundary.

The model may draft only:

- title,
- summary,
- findings,
- approved evidence references.

The application still owns every allocation and publication decision. Model output is validated against a strict Pydantic schema, checked against an application-owned evidence allowlist, and then validated again through the typed `WorkProduct` contract. Invalid output fails closed into the existing bounded recovery path.

The live production path uses Hugging Face Inference Providers through the OpenAI-compatible router. Runtime model selection is configuration rather than auction policy.

> **The auction grants authority to execute. The handler produces content. The application owns lineage and validates the result.**

## Gradio Demo

**Live Space:** https://huggingface.co/spaces/FlyingNunchucks/11-distributed-auction-task-allocation-agent

The business-first demo exposes:

- **Task Marketplace** — task winners, scores, bid counts, rounds, messages, and execution cost,
- **Centralized vs Distributed** — controlled architecture tradeoff,
- **Stress Evaluation** — nine deterministic scenarios,
- **Work Products** — validated outputs from winning peers,
- **Protocol Audit** — append-only allocation events,
- **Engineering Boundary** — what the LLM may and may not control.

Deterministic mode is the safe default. LLM-assisted mode changes the substantive work products after award but does not control eligibility, bidding admission, scoring, settlement, winner selection, reauction, or publication.

## Production Validation

Final automated test result:

```text
140 passed
```

The public Hugging Face Space was manually validated in both:

- deterministic mode,
- live LLM-assisted mode.

The deterministic production check completed with:

- 5/5 tasks completed,
- 50 protocol messages,
- $347 synthetic execution cost,
- 1.00 allocation efficiency,
- and the expected winner path: Atlas → Ledger → Sentinel → Veritas → Quill.

The live LLM-assisted production check also completed successfully while preserving the same application-controlled allocation boundary. The model changed the awarded work-product content, not the auction authority.

## Reusable Primitive

> **A typed distributed task-auction protocol in which peers locally evaluate work, submit validated bids, and deterministically allocate and reallocate tasks using capability, confidence, cost, and availability without a central semantic planner.**

## Scope and Known Limitations

This project deliberately does **not** implement:

- malicious or deceptive bidding,
- bidder collusion,
- long-term reputation or trust scoring,
- Byzantine-agent detection,
- compromised-agent isolation,
- redundant execution,
- network-partition recovery,
- broad self-healing,
- or general fault-tolerant replanning.

The current auction assumes a shared deterministic utility policy and authoritative application-owned capability profiles. Those limits are intentional so Agent 11 teaches decentralized allocation without quietly becoming the later fault-tolerance and trust agents.

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
- `MODEL_ID` — Space variable identifying the inference model.
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
