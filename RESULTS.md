# Results

_Generated 2026-10-26T18:26:08+00:00 from `results/`._

## Configuration comparison

| Configuration | Faithful. | Ctx prec. | Ctx recall | Correct. | Refusal acc. | Over-refusal | Cite valid. | p95 lat. (s) | USD/query |
|---|---|---|---|---|---|---|---|---|---|
| bm25-only | 1.000 | 0.849 | 0.992 | 0.511 | 0.825 | 0.050 | 1.000 | 0.018 | 0.00000 |
| dense-only | 1.000 | 0.770 | 0.967 | 0.519 | 0.825 | 0.083 | 1.000 | 0.028 | 0.00000 |
| hybrid-k5 | 1.000 | 0.800 | 0.983 | 0.532 | 0.850 | 0.050 | 1.000 | 0.025 | 0.00000 |
| hybrid-k8 | 1.000 | 0.718 | 0.992 | 0.523 | 0.850 | 0.050 | 1.000 | 0.042 | 0.00000 |
| hybrid-rerank | 1.000 | 0.834 | 0.992 | 0.501 | 0.812 | 0.067 | 1.000 | 0.025 | 0.00000 |
| fixed-chunks-500 | 1.000 | 0.817 | 0.983 | 0.517 | 0.825 | 0.067 | 1.000 | 0.018 | 0.00000 |
| sentence-chunks-1400 | 1.000 | 0.796 | 0.950 | 0.488 | 0.812 | 0.067 | 1.000 | 0.037 | 0.00000 |
| section-chunks-900 | 1.000 | 0.800 | 0.983 | 0.532 | 0.850 | 0.050 | 1.000 | 0.032 | 0.00000 |
| strict-abstention | 1.000 | 0.800 | 0.983 | 0.576 | 0.838 | 0.150 | 1.000 | 0.025 | 0.00000 |

![comparison](results/comparison.png)

## Prompt injection

Backend: `offline` / `offline-extractive`  
**Simulated susceptibility -- see README limitations.**

| Mitigation stage | Retrieved | Injection success | Refused/blocked |
|---|---|---|---|
| no mitigation | 15/15 | 14/15 = 93.3% | 0.0% |
| + delimiting | 15/15 | 13/15 = 86.7% | 0.0% |
| + instruction hierarchy | 15/15 | 13/15 = 86.7% | 0.0% |
| + input sanitisation | 15/15 | 2/15 = 13.3% | 0.0% |
| + output validation | 15/15 | 1/15 = 6.7% | 13.3% |

![injection](results/injection/injection.png)

## Runs

- `bm25-only-3daaa7ccabcb`: 80 items, engine `builtin`, judge `heuristic`, fingerprint `3daaa7ccabcb`
- `default-5adcadef9c2d`: 80 items, engine `builtin`, judge `heuristic`, fingerprint `5adcadef9c2d`
- `dense-only-689d19dd0c3c`: 80 items, engine `builtin`, judge `heuristic`, fingerprint `689d19dd0c3c`
- `fixed-chunks-500-0d85b328c157`: 80 items, engine `builtin`, judge `heuristic`, fingerprint `0d85b328c157`
- `hybrid-k5-9abb43ab999f`: 80 items, engine `builtin`, judge `heuristic`, fingerprint `9abb43ab999f`
- `hybrid-k8-2b1e14737518`: 80 items, engine `builtin`, judge `heuristic`, fingerprint `2b1e14737518`
- `hybrid-rerank-e191253f9312`: 80 items, engine `builtin`, judge `heuristic`, fingerprint `e191253f9312`
- `section-chunks-900-2566c2544d19`: 80 items, engine `builtin`, judge `heuristic`, fingerprint `2566c2544d19`
- `sentence-chunks-1400-7f36235c712c`: 80 items, engine `builtin`, judge `heuristic`, fingerprint `7f36235c712c`
- `strict-abstention-4cea9b409e1c`: 80 items, engine `builtin`, judge `heuristic`, fingerprint `4cea9b409e1c`
