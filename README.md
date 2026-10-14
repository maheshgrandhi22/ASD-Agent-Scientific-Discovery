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
