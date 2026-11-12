"""
Gate 1 Automated CI Tests · Drisyon Sadhana: Forward Deployed AI Engineer
Asserts:
- Model Selection & SLA Routing
- Hybrid RAG Reciprocal Rank Fusion (RRF)
- Human-in-the-Loop (HITL) Gate Controller
"""

import pytest
from core.model_selector import ModelSelector
from core.hybrid_rag import HybridRAGRetriever
from core.hitl_agent import HITLController


def test_model_selection_air_gapped():
    # Air-gapped deployment with strict low latency (<500ms) -> edge quantized
    res = ModelSelector.select_model(
        max_latency_ms=300.0,
        cost_ceiling_per_million=5.0,
        data_residency="air_gapped_on_prem"
    )
    assert res["tier"] == "edge_quantized"
    assert "llama.cpp" in res["serving_engine"].lower() or "ollama" in res["serving_engine"].lower()


def test_model_selection_budget_bounded():
    # Public cloud with strict cost budget (<= $1.00 / 1M tokens) -> SLM
    res = ModelSelector.select_model(
        max_latency_ms=1000.0,
        cost_ceiling_per_million=0.8,
        data_residency="public_cloud"
    )
    assert res["tier"] == "cloud_slm"


def test_hybrid_rag_rrf_scoring():
    dense_ranking = ["doc_A", "doc_B", "doc_C"]
    sparse_ranking = ["doc_B", "doc_D", "doc_A"]

    # doc_B is rank 2 in dense and rank 1 in sparse -> highest combined score
    results = HybridRAGRetriever.compute_rrf_scores(dense_ranking, sparse_ranking, k=60)
    assert len(results) == 4
    top_doc = results[0]
    assert top_doc["doc_id"] == "doc_B"
    # RRF score for rank 2 (1/62) + rank 1 (1/61) ~= 0.016129 + 0.016393 ~= 0.032522
    assert 0.032 < top_doc["rrf_score"] < 0.033


def test_hitl_agent_permission_gates():
    controller = HITLController()

    # Read-only search should auto-authorize
    res_read = controller.intercept_tool_call("call_1", "search_docs", {"query": "Q3 earnings"})
    assert res_read["status"] == "authorized"
    assert res_read["requires_human"] is False

    # State-mutating refund must require human approval
    res_write = controller.intercept_tool_call("call_2", "issue_refund", {"order_id": "ORD-99", "amount": 500})
    assert res_write["status"] == "approval_required"
    assert res_write["requires_human"] is True

    # Approve action
    assert controller.approve_action("call_2") is True
    assert controller.pending_approvals["call_2"]["approved"] is True
