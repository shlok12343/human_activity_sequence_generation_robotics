# Kitchen Activity Sequence Generation and DAG Visualization

This project generates kitchen activity sequences with LLMs, derives affordance rules, and visualizes sequence flow as merged DAGs.

It is designed for kitchen safety workflows where you want:
- normal/alternate task sequences,
- hazardous variants,
- inferred affordance constraints from observed actions,
- graph outputs to compare behavior paths.

## What This Project Does

- Generates object-centric activity sequences using LangChain + Gemini (`kitchen_activity_sequence_generator.py`).
- Generates hazardous-only sequences for safety analysis (`generate_hazardous_sequences` in the same file).
- Generates affordance rules from a base sequence (`affordance_generator.py`).
- Builds and renders process DAGs with merged prefixes and converged suffix states (`build_process_dag.py`).
- Supports colored edge DAGs for normal vs hazardous transitions (black vs red).

## Project Structure

- `kitchen_activity_sequence_generator.py` - main sequence generation workflows.
- `affordance_generator.py` - affordance rule generation from a given sequence.
- `build_process_dag.py` - DAG construction and Graphviz rendering helpers.
- `hazard_analyzer.py` - additional hazard-focused analysis utilities.
- `requirements.txt` - Python dependencies.
- `.env` - local API key configuration (git-ignored).

## Requirements

- Python 3.12+ recommended
- Gemini API key
- Graphviz system binary (`dot`) installed

## Setup

1) Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Install Graphviz system package (needed for PNG rendering):

```bash
brew install graphviz
```

4) Create `.env` in project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Usage

### 1) Generate Affordance Rules

Run:

```bash
python affordance_generator.py
```

This prints a structured list of inferred affordance rules for the test sequence in `main()`.

### 2) Generate Activity Sequences + Build DAG

Run:

```bash
python kitchen_activity_sequence_generator.py
```

By default, this script:
- generates sequences for the currently selected object setup in `main()`,
- prints each generated sequence,
- renders a DAG image (for example `process_dag.png`).

### 3) Build DAG from Existing Sequences

If you want only DAG rendering from predefined sequence lists:

```bash
python build_process_dag.py
```

## Core Functions

- `generate_sequences_object(...)`  
  Generates mixed sequence variants (efficient, alternate-order, edge cases, etc.).

- `generate_hazardous_sequences(...)`  
  Generates sequences that are explicitly hazardous.

- `generate_affordance_rules(...)`  
  Produces affordance constraints from an observed sequence.

- `build_and_render_process_dag(...)`  
  Creates a merged DAG and outputs a PNG via Graphviz `dot`.

- `build_and_render_process_dag_with_colored_edges(...)`  
  Renders a comparison DAG with normal/hazardous edge coloring.

## Notes

- Keep `.env` private; it is listed in `.gitignore`.
- If rendering fails with `ExecutableNotFound: dot`, install Graphviz with Homebrew and rerun.
- You can customize object scenarios by editing the base sequences and affordance rules inside `kitchen_activity_sequence_generator.py`.

