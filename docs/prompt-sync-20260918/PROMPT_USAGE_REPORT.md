# Robot prompt text usage after NavProbe synchronization

Robot baseline: `98f050a1bd5fb9ee774269e4c15b86037c44393b`. Upstream: `bfa2805b219bfe7e0e0f721ed9bb0a47f9049f61`.

## Method

Both versions are rendered with their actual prompt constructors and identical placeholder inputs: default VLN, dynamic agenda, FSS, four view captions, and the conditional modes below. The script asserts that the imported runtime belongs to this repository. Full text fixtures, flags, source hashes, and component counts are in [prompt_usage.json](prompt_usage.json).

Tokens use **tiktoken 0.14.0, `o200k_base`**. Characters are Unicode code points, including whitespace. System and individual user text blocks are counted separately and summed; all measured system strings are unchanged.

Counts include the fixed placeholder task/memory payloads. Excluded: image tokens, API envelopes, generated/reasoning tokens, retries, and variable production history. These are local text counts, not API billing or measured episode usage. The variants are alternatives and must not be summed as an episode total. No robot or live model is called.

## Results

| Variant | Characters before | After | Delta | Tokens before | After | Delta | Token change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initializer | 1,036 | 1,269 | +233 | 170 | 210 | +40 | +23.5% |
| Executive: regular | 10,832 | 9,392 | -1,440 | 2,165 | 1,925 | -240 | -11.1% |
| Executive: first step, budget 6 but retrieve disabled | 10,107 | 8,702 | -1,405 | 2,016 | 1,777 | -239 | -11.9% |
| Executive: after retrieval, 1 of 6 rounds | 11,184 | 9,744 | -1,440 | 2,233 | 1,993 | -240 | -10.7% |
| Executive: anchor, budget exhausted | 10,843 | 9,893 | -950 | 2,156 | 1,995 | -161 | -7.5% |
| Executive: terminal, retrieve available | 12,885 | 12,566 | -319 | 2,568 | 2,530 | -38 | -1.5% |
| Executive: terminal after final retrieval | 11,984 | 11,464 | -520 | 2,375 | 2,293 | -82 | -3.5% |
| Skill: physical reference | 6,279 | 6,092 | -187 | 1,250 | 1,207 | -43 | -3.4% |
| Skill: first step, no anchors | 5,752 | 5,404 | -348 | 1,137 | 1,066 | -71 | -6.2% |
| Skill: historical reference | 7,015 | 7,523 | +508 | 1,390 | 1,466 | +76 | +5.5% |

## Interpretation

This comparison includes P0 contract/semantic repairs and P1.1/P1.2 retrieval guidance/compression together. Most variants shrink; initialization and historical-reference skill prompts grow because they now preserve instruction semantics and explain physical execution and historical `stay` accurately. Compression does not imply improved navigation or stronger-model compatibility without new rollouts.

Other runtime modules are unchanged, so their fixed-text deltas are zero under matched inputs. The P1.3 handoff audit introduced no additional context packet. Actual dynamic inputs and episode token totals may change as decisions change.

## Reproduction

Use the matching runtime source identified by `after_sha256`, the declared robot dependencies, and optional `tiktoken==0.14.0` with the `o200k_base` vocabulary available. No tokenizer dependency was added to the robot runtime.

```bash
python docs/prompt-sync-20260918/measure_prompt_usage.py --before-ref 98f050a1bd5fb9ee774269e4c15b86037c44393b
```
