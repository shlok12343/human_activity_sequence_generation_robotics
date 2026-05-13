"""
Open-world kitchen hazard prototype: discover object interactions via Gemini,
reuse ``run_state_graph_pipeline`` for interaction-object states, and judge
(original × interaction × state) combinations for hazards.

Run from repository root::

    export GOOGLE_API_KEY=...
    python -m soda_can_problem.cli --help
"""
