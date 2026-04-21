## 1. Interpreting the curves like a researcher

### Core takeaway
The run does **not** look like smooth linear learning. It looks like **staged competence acquisition** with plateaus and jumps.

### Phase A: Early compression (5m → 15m)
The model first learns cheap wins:
- common token frequencies
- punctuation habits
- sentence starts/stops
- generic English rhythm

This can improve loss quickly while outputs remain deranged.

### Phase B: Local structure emergence (15m → 1h)
Then it starts learning:
- phrase chunks
- noun phrase patterns
- common continuations
- basic syntax persistence

This is where outputs become grammatical nonsense rather than raw sludge.

### Phase C: Semantic mimicry layer (1h → 8h)
Now it learns discourse genres:
- explanations
- philosophy tone
- scientific taxonomy tone
- technical manual tone
- narrative paragraph form

This is where many of the funniest artifacts appear: strong structure with broken grounding.

### Phase D: Control gains / cleanup (8h → 12h)
Later improvements show up as:
- less repetition
- longer coherent spans
- fewer collapse loops
- better token choice under uncertainty

That matches the stronger 12h metrics.

### Why jumps happen instead of smooth curves
Loss curves can look smooth while qualitative behavior changes in steps.

Many visible behaviors need several subskills at once:
- syntax + memory + token calibration
- discourse template + local consistency
- repetition suppression + vocabulary spread

Once thresholds are crossed, outputs can suddenly look much better.

### Why earlier checkpoints can sometimes beat later ones on a metric
There are tradeoffs:
- more diversity can reduce local coherence
- lower repetition can increase weird novelty
- better compression does not always mean nicer prose

So no single metric should be trusted alone.

### Working interpretation
The tiny model appears to move from:

**statistical text imitator → weak genre model → early coherent generator**

That is a real developmental story.

### Bottom line
This looks less like steady improvement and more like **layered skill acquisition** becoming visible at different checkpoints.

