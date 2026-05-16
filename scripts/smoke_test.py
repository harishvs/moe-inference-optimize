"""End-to-end sanity check: load OLMoE under vLLM and generate one token.

Run this first to confirm the GPU, drivers, vLLM install, and weight cache
are all wired up. Reports rough TTFT just so you have something to compare
against the proper baseline benchmark.
"""
from __future__ import annotations

import time

from vllm import LLM, SamplingParams

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
PROMPT = "Write a Python function that returns the first n Fibonacci numbers."


def main() -> None:
    llm = LLM(model=MODEL)
    t0 = time.perf_counter()
    outputs = llm.generate([PROMPT], SamplingParams(max_tokens=1))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    text = outputs[0].outputs[0].text
    print(f"output: {text!r}")
    print(f"first-token latency (incl. scheduling overhead): {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    main()
