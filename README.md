# Drisyon Sadhana · Forward Deployed AI Engineer Starter Kit (`fde40`)

Welcome to the **Forward Deployed AI Engineer (40-Hour Track)** starter repository.

This lab kit contains the reference modules, test harnesses, and automated GitHub Actions CI gates required for your Sadhana Kosha portfolio.

## Structure
```
drisyon-sadhana-fde40/
├── core/
│   ├── model_selector.py     # Customer SLA routing & latency/cost evaluation
│   ├── hybrid_rag.py         # Reciprocal Rank Fusion (RRF) for dirty customer data
│   └── hitl_agent.py         # Human-in-the-Loop gates on mutating tool calls
├── tests/
│   └── test_gate1.py         # Automated Gate 1 test suite
├── .github/
│   └── workflows/
│       └── gate.yml          # GitHub Actions automated test gate
├── pytest.ini
└── requirements.txt
```

## Local Setup & Gate Verification

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Gate 1 Verification
pytest tests/test_gate1.py -v
```

Once all tests turn green, push your code to your GitHub repository to trigger the automated CI Gate.
