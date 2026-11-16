# Autonomous Multimodal Scientific Discovery Agent (ASD-Agent)

## Overview

The Autonomous Multimodal Scientific Discovery Agent is a research-oriented AI project designed to support scientific discovery through research question validation, structured planning, and scientific literature retrieval.

The project is being developed incrementally, with a focus on building a reliable foundation for an autonomous scientific research workflow.

## Current Features

- **Research Question Validation:** Checks that research questions are valid, non-empty strings within the permitted length.
- **Scientific Research Planning:** Generates research objectives, subquestions, proposed methods, and next steps.
- **Scientific Literature Retrieval:** Searches arXiv for relevant scientific papers.
- **Modular Architecture:** Organizes the application into separate agent components.
- **Automated Testing:** Includes unit tests for key components.

## Technology Stack

- Python
- Requests
- arXiv API
- unittest
- Git and GitHub

## Project Structure

```text
autonomous-scientific-discovery-agent/
├── README.md
├── requirements.txt
├── input_validator.py
├── src/
│   ├── main.py
│   └── agent/
│       ├── discovery_agent.py
│       ├── research_planner.py
│       └── literature_retriever.py
└── tests/
    ├── test_input_validator.py
    └── test_research_planner.py
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/maheshgrandhi22/ASD-Agent-Scientific-Discovery.git
cd ASD-Agent-Scientific-Discovery
```

### 2. Create a virtual environment

```bash
py -m venv .venv
```

### 3. Activate the virtual environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
py -m pip install -r requirements.txt
```

## Run the Application

From the project root directory, run:

```bash
py -m src.main
```

Enter a research question when prompted, for example:

```text
How can AI improve drug discovery?
```

## Run Tests

Run the test suite with:

```bash
py -m unittest discover -s tests -v
```

## Development Progress

- Day 1: Initial project setup.
- Day 2: Scientific discovery agent foundation.
- Day 3: Research question validation.
- Day 4: Validation integration and unit tests.
- Day 5: Structured scientific research planning.
- Day 6: Scientific literature retrieval.

## Future Development

Planned improvements include richer literature analysis, evidence extraction, research synthesis, and further development of the autonomous scientific discovery workflow.

## Project Goal

To develop a modular AI research agent that can help researchers explore scientific questions, discover relevant literature, and progressively support evidence-based scientific investigation.


## Peer-Review Evidence Verification

The discovery workflow uses a fail-closed publication-verification policy.
ArXiv or PubMed presence, a DOI, or generic bibliographic metadata alone is
not treated as proof that an article completed peer review.

The default provider checks Crossref for a matching journal-article DOI and
title, then looks for Crossref-registered, DOI-linked pre-publication referee
reports and an editor report with an explicit `accept` recommendation. Only
when all required records are present does the workflow allow the article into
literature analysis, the vector store, and hypothesis generation. Papers with
missing identifiers, missing review records, mismatched metadata, or provider
errors remain unverified and are excluded.

**Important limitation:** Crossref makes publisher/member-deposited metadata
available; it does not independently audit the truth of every deposited
record. Crossref review records are positive documentary evidence that a
review process was registered, not proof that the underlying research is
correct or that every editorial process was genuine. Review metadata is not
registered for every journal or article. Therefore, absence of Crossref review
records means **unverified by this strict provider**, not fake or necessarily
unreviewed. The current policy may exclude legitimate peer-reviewed papers
whose publishers do not deposit review records.

The provider requires a DOI. PMID-only records cannot pass this Crossref
provider and remain unverified. Network errors and incomplete API responses
also fail closed. The fake provider used in unit tests is test-only and must
never be configured in production.

The multi-source retriever extracts DOI metadata when supplied by arXiv and
prioritises DOI-bearing records when applying the combined result limit. This
prevents DOI-less arXiv results from consuming the entire small result budget
before DOI-bearing PubMed records can be considered. DOI presence is only a
candidate-selection aid; it is not peer-review verification.

Peer-review evidence verification does not detect fabricated data, flawed
methods, plagiarism, or false conclusions. Those require separate research
integrity and scientific-quality checks.

## Non-destructive integration smoke checks

To check the live Crossref REST API and ChromaDB metadata filtering in the
Windows environment, run this from the project root:

```powershell
py scripts/verify_external_integrations.py
```

The script makes a read-only Crossref request (defaulting to a DOI shown in
Crossref's REST API quick-start documentation) and creates a temporary
ChromaDB database that is deleted automatically when the script exits. It
never opens or modifies the project's persistent `data/vector_db` directory.

A successful Crossref metadata request proves API connectivity and DOI
metadata retrieval only. The separate `strict_peer_review_result` must also
report `peer_reviewed: true` for the selected DOI to pass this project's
current strict Crossref evidence policy. A false result means the required
Crossref review records were not found or did not meet the policy; it does not
prove that the article is fake or was never peer reviewed.

A successful ChromaDB smoke check verifies that the installed ChromaDB version
supports persistent collections and the metadata filter used to exclude
unverified and legacy records. It is an integration smoke check, not a full
replacement for the project's `tests/test_vector_store.py` suite.
