# Self-Correction Data Generation

Pipeline that produces two SFT datasets (D1 and D2) for teaching an LLM to self-correct mathematical reasoning errors. Built on MetaMathQA, using Mistral-7B as the step generator and DeepSeek-R1 as the judge / supervision generator.

---

## Error step generation (shared upstream)

For each MetaMathQA example:

1. Sample a gold prefix from the gold solution (1 to half of the steps).
2. Use Mistral-7B (T=1.2) to draft a candidate next step after the prefix.
3. Generate 8 rollouts from that candidate (T=0.8).
4. Verify rollout correctness with symbolic answer extraction; fall back to a DeepSeek-R1 judge.
5. Label the candidate **wrong** only if **all 8 rollouts produce wrong final answers**.

The all-rollouts-wrong filter selects steps where the base model has *no* recovery path on its own — supervision is non-redundant. A DeepSeek-R1 judge then attributes each wrong step to a single source from `{step_1, step_2, …, question, independent}` — which prior input the model misused when generating the wrong step.

---

## Dataset 1 — single-step self-correction

The model is trained to detect and correct an error **at the wrong step itself**.

For each `(prefix, wrong_step, attribution)` triple, DeepSeek-R1 generates an *error trace* (first-person mid-generation pivot, opener keyed on the attribution source — e.g. "Let me recheck Step 2"), a one-sentence *diagnosis*, and a *corrected step*.

```
USER:      Problem: …\n\nSolve step by step.
ASSISTANT: {gold prefix}        ← loss computed
           {wrong step}         ← loss MASKED
           Error trace: …       ← loss computed
           Diagnosis: …         ← loss computed
           Corrected step: …    ← loss computed
           {continuation to #### N}
```

---

## Dataset 2 — multi-step late detection

The model is trained to detect an error **several steps after** the actual mistake, retrace to the origin, and correct from there.

Each downstream rollout step is first labeled `propagated` (silently inherits a wrong value, no new mistake) or `new_error` (fresh mistake compounded on top). A detection point is then **randomly sampled** from any downstream step ≥ 1 step after the wrong step — forcing late detection, the realistic deployment scenario. DeepSeek-R1 produces *detection*, *retrace*, *error trace*, *diagnosis*, and a *correction* chain from the origin step through the detection step.

```
USER:      Problem: …\n\nSolve step by step.
ASSISTANT: {gold prefix}                          ← loss computed
           {wrong step + downstream to detection} ← loss MASKED (single span)
           Detection: … / Retrace: …
           Error trace: … / Diagnosis: …
           Correction: …                          ← loss computed
           {continuation to #### N}
```
