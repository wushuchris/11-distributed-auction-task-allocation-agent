# Agent 11 — Distributed Auction Task Allocation Agent

Distributed multi-agent task allocation using typed auctions, deterministic bidding rules, bounded reallocation, and auditable coordination.

## Status

**Work in progress.** The repository currently contains only the initial project scaffold. Auction schemas and behavior will be added incrementally with automated tests.

## Problem

When several autonomous peers are capable of performing the same task, a central manager should not have to decide every assignment manually. This project explores a bounded task marketplace where peers independently decide whether to bid and a deterministic protocol selects the winner.

## Engineering Principle

> **Peers decide whether to compete. The protocol decides who wins.**

The eventual system will keep allocation authority in application code rather than model prose. Models may later contribute bounded task work, but they will not control identity, capability registration, bid validation, auction settlement, or publication boundaries.

## Planned Learning Objective

Agent 11 builds on peer-to-peer coordination by adding:

- typed task announcements,
- typed bids and abstentions,
- capability, confidence, availability, and cost scoring,
- deterministic winner selection,
- bounded reauctioning after explicit failure,
- allocation-efficiency metrics,
- message-complexity measurement,
- and a centralized comparison baseline.

## Planned Synthetic Scenario

The public demo will use a fictional corporate due-diligence mission with synthetic, public-safe data. Multiple peers will have intentionally overlapping capabilities so the auction has a real allocation decision to make.

No private curriculum material, client data, financial data, personal information, or real credentials belong in this repository or its demo.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── auction_coordination/
│       └── __init__.py
└── tests/
    └── test_scaffold.py
```

## Development Approach

The build will proceed incrementally:

1. typed schemas,
2. local bidding policy,
3. auction protocol and validation,
4. deterministic scoring and settlement,
5. bounded reauctioning,
6. deterministic work handlers,
7. evaluation and centralized comparison,
8. optional bounded LLM work handlers,
9. business-first demo UI,
10. test-gated deployment.

## License

MIT
