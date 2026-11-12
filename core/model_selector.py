"""
Drisyon Sadhana · Forward Deployed AI Engineer Starter Kit
Module 1: Customer SLA & Model Selection Matrix
"""

from typing import Dict, Any, Literal


class ModelSelector:
    """Selects optimal model tier based on customer latency, budget, and data residency constraints."""

    @staticmethod
    def select_model(
        max_latency_ms: float,
        cost_ceiling_per_million: float,
        data_residency: Literal["public_cloud", "private_vpc", "air_gapped_on_prem"]
    ) -> Dict[str, Any]:
        """
        Evaluates customer constraints and routes to:
        - frontier_cloud (e.g. Claude 3.7 / GPT-4o)
        - cloud_slm (e.g. Claude 3.5 Haiku / GPT-4o-mini)
        - on_prem_open_weights (e.g. Llama-3-70B / Qwen-2.5 on vLLM)
        - edge_quantized (e.g. Llama-3-8B INT4 on Ollama/llama.cpp)
        """
        # Hard Rule 1: Air-gapped on-premise requires local open-weights
        if data_residency == "air_gapped_on_prem":
            if max_latency_ms < 500:
                return {
                    "tier": "edge_quantized",
                    "model": "Llama-3.1-8B-Instruct-GGUF-Q4_K_M",
                    "serving_engine": "llama.cpp / Ollama",
                    "rationale": "Ultra-low latency air-gapped on-prem deployment."
                }
            return {
                "tier": "on_prem_open_weights",
                "model": "Llama-3.3-70B-Instruct-AWQ",
                "serving_engine": "vLLM with PagedAttention",
                "rationale": "High-accuracy air-gapped server cluster."
            }

        # Rule 2: Budget constraints (< $1.00 / 1M tokens)
        if cost_ceiling_per_million <= 1.0:
            return {
                "tier": "cloud_slm",
                "model": "gpt-4o-mini / claude-3-5-haiku",
                "serving_engine": "Managed Provider API with Prompt Caching",
                "rationale": "Cost-bounded high-throughput cloud deployment."
            }

        # Rule 3: Complex reasoning with flexible budget
        return {
            "tier": "frontier_cloud",
            "model": "claude-3-7-sonnet / gpt-4o",
            "serving_engine": "Managed Provider API",
            "rationale": "Maximum reasoning capacity for mission-critical enterprise workflows."
        }
