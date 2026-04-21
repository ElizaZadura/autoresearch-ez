## 3. How legit these metrics are vs misleading

### Short answer
The metrics are **useful directional signals**, but they are not strong enough to stand alone. They become most valuable when read together with the prompt outputs.

### What looks legitimately useful

#### 1. `val_bpb`
This is the strongest quantitative metric in the set.

Why it matters:
- it reflects compression/predictive improvement on held-out text
- it tracks training progress better than subjective reading alone
- it gives a comparable curve across checkpoints

Limits:
- lower `val_bpb` does **not** guarantee better generations on every prompt
- it can improve while outputs still look absurd
- it does not tell you what kind of capability improved

### 2. Repetition metrics (`rep_onset_word`, `clean_span`, `rep_rate`)
These are useful because repetition collapse is one of the most visible failure modes in small models.

Why they help:
- they capture when token attractor loops dominate
- they make “shingles mode” and “instructions mode” visible numerically
- they help compare checkpoints beyond pure intuition

Limits:
- they reward avoiding repetition, but not meaning
- weird but diverse nonsense can score better than dull but coherent text
- repeated motifs are not always bad if they are contextually justified

### 3. Sentence completion rate
This is a decent proxy for syntactic tidiness.

Why it helps:
- it distinguishes raw collapse from sentence-shaped output
- it is easy to compare across checkpoints

Limits:
- it can be gamed by fluent nonsense
- complete sentences are not the same as coherent thought

### 4. Type-token ratio (TTR)
Useful, but fragile.

Why it helps:
- gives some signal about lexical diversity
- can reveal overreliance on a small token set

Limits:
- highly length-sensitive
- can make nonsense look better if the nonsense is varied
- not reliable as a primary quality metric

### 5. Average sentence length
Mostly descriptive, not evaluative.

Why it helps:
- can expose abrupt style shifts or run-on behavior
- useful for comparing checkpoints loosely

Limits:
- longer is not better
- shorter is not better
- easily distorted by one bad paragraph

### What is misleading if over-trusted

#### 1. Any single metric alone
No one number captures coherence, prompt adherence, interestingness, or semantic stability.

#### 2. “Cleaner” automatically meaning “better”
A later checkpoint may become less repetitive but also less surprising.

#### 3. Treating proxies as ground truth
These metrics are proxies for visible behaviors, not direct measurements of understanding.

### Best reading strategy
Use three layers together:

1. **val_bpb** for overall training progress
2. **text metrics** for surface failure modes and stability
3. **manual prompt review** for actual qualitative change

That combination is much more trustworthy than any one layer.

### My working verdict
- `val_bpb` is legit and important
- repetition metrics are useful and informative
- sentence completion and TTR are secondary support metrics
- manual review is still necessary

### Bottom line
These metrics are not fake, but they are also not self-sufficient. They are best treated as **instruments**, not verdicts.

