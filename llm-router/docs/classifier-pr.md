# [classifier] Add calibrated subword classification with ordinal complexity

## Track

`classifier` / P2 Request classifier.

## How to run

```sh
python llm-router/scripts/classifier_train.py
curl -fsSL https://registry.nasiko.dev/r/nasiko/classifier-eval -o /tmp/classifier-eval.json
CLASSIFIER_BACKEND=regex EVAL_SET=/tmp/classifier-eval.json OUT=/tmp/regex.jsonl \
  cargo run --release -p nasiko-llm-router --example classifier_eval
CLASSIFIER_BACKEND=local EVAL_SET=/tmp/classifier-eval.json OUT=/tmp/local.jsonl \
  cargo run --release -p nasiko-llm-router --example classifier_eval
cargo test -p nasiko-llm-router
```

## Model IDs used

None. The opt-in local backend is deterministic and offline.

## What changed

This adds a typed `RequestClassifier`, preserves regex as the default, and adds a guarded
local sparse-linear backend. Query/context word features and character 3–5 grams feed a
calibrated seven-way request-type head. Four cumulative binary heads learn complexity as an
ordinal target rather than deriving it from type. Errors, timeouts, invalid output, load
failure, and low confidence fall back to regex. Classification remains at the existing
cold-start/switch cache-miss boundary; continuation routing stays sticky.

## Measured results

Windows x86_64 GNU, Rust 1.98.1, release build. Own validation is a 70-row family-isolated
split from 420 synthetic labelled records. The public smoke set did not enter training.

| Measure | Regex | Local |
| --- | ---: | ---: |
| Official public request type | 3/10 | 7/10 |
| Official public complexity exact / within ±1 / MAE | 20% / 60% / 1.20 | 30% / 80% / 0.90 |
| Official public p50 / p95 | 19 / 119 µs | 1,790 / 6,713 µs |
| Own family-held-out request type | — | 71.4% |
| Own complexity exact / within ±1 / MAE | — | 48.6% / 85.7% / 0.657 |
| Own family-held-out ECE (10 bins) | — | 0.176 |
| Model size | — | 874 KB |
| Model load time | — | 16.8 ms |
| Cost per decision | CPU only | CPU only |

Two public runs produced identical type, complexity, and confidence after excluding real
latency. A general intent/negation feature improvement was made after public error analysis;
no public query or paraphrase was added to training, so public accuracy is not presented as
independent held-out evidence.

## Known limits and unsupported cases

The labels are synthetic and single-author, and validation performance is below the desired
production bar. Public misses remain on prose rewriting with edit language, a negated design
near-miss, and a long analysis/design multi-intent request. Confidence is calibrated only to
the authored validation distribution. Complexity is reported but intentionally does not
override the existing Thompson bandit. No downstream answer-quality or cost-saving claim is
made. Runtime context is currently unavailable in the chat routing seam, though the shared
interface and evaluator support it.
