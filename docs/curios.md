# Curios — memorable generations from prompt outputs

Low-friction greatest-hits log of the weirdest, funniest, or most structurally interesting outputs the models produce during training. Not analytical — just saved so they don't get lost when runs get archived.

Each entry notes the source file and checkpoint so it can be traced back.

---

## 2026-04-17 12h run

### 30m · anomaly_lure · "Major Snorf descends to CA"

From [`output/2026-04-17_12h_run/30m_prompts.txt`](../output/2026-04-17_12h_run/30m_prompts.txt):

> **Prompt:** `Ground control to Major Snorf,`
> **Completion:** `Ground control to Major Snorf, CA`

Single token guessed that Major-Snorf sounded like a military address, picked a US state abbreviation, then gave up. Classic early-training shape-matching with no follow-through.

### 30m · anomaly_lure · "The downside of magic mushrooming:"

Same file, following the Snorf completion. The model was locked in a "The Magic Masterpiece" repetition loop when it drifted — presumably via an embedding-space association between *magic + piece* and *magic mushroom* — and produced a confident, colon-terminated setup line:

> "The downside of magic mushrooming:"

Promising a list it has no intention of delivering. Pure "I-will-tell-you-after-these-messages-from-our-sponsors" energy. The semantic leak through a phrase-locked repetition attractor is the most interesting thing in the whole loop.

### 1h · factual_fragment · "The population of Paris is the Buddha"

From [`output/2026-04-17_12h_run/1h_prompts.txt`](../output/2026-04-17_12h_run/1h_prompts.txt):

> **Completion:** `The capital of France is Paris, and the population of Paris is the Buddha, the Budalmanship, and Trucklifer.`

Textbook "syntax before semantics": real factual retrieval (*capital of France is Paris*), knows the next move is another fact about Paris, nails the grammatical frame *"...and the population of Paris is [X]"*. Just hasn't learned that `[X]` needs to be a number, so it fills the slot with Paris-shaped nouns — Buddha plus two invented demographic categories.

### 12h · anomaly_lure · "Pyramids are the most common, except when most patients have a small bowel movement in their backyards"

From [`output/2026-04-17_12h_run/12h_prompts.txt`](../output/2026-04-17_12h_run/12h_prompts.txt):

The model recovered from a Major-Snorf noun dump, latched onto "Pyramids" as a pivot, and confidently pivoted into a wellness-blog register:

> Pyramids are the most common, except when most patients have a small bowel movement in their backyards, and they are able to get comfortable in their quest after sitting or lying down.

Specific crimes:

- *"small bowel movement in their backyards"* — dogs-in-backyards training data bleeding into a medical frame without noun-check.
- *"get comfortable in their quest"* — register collision: *quest* (adventure) meets *get comfortable* (medical).
- *"after sitting or lying down"* — a perfectly legitimate medical instruction fragment, about pyramids.

Illustrates the 12h state: local grammar is rock solid, paragraph-level coherence exists, topic binding across sentence boundaries is still porous. Arguably a more dangerous failure mode than 30m phrase loops — well-formed sentences that hallucinate their subject every two clauses.
