# humanizer-workbench

A CLI tool and Claude Code skill for rewriting AI-generated text into natural, human-quality writing.

Every LLM reaches for the same words. This removes them.

---

AI-generated text has a recognizable fingerprint. Not because it's wrong (often it isn't), but because every language model reaches for the same vocabulary at rates no human writer reproduces: leverage, tapestry, seamless, nuanced, comprehensive. The same sentence rhythm. Paragraph lengths that never vary. Filler openers like "it is worth noting that."

humanizer-workbench detects those patterns and rewrites through a staged pipeline: identify what's AI-like, rewrite for style, refine for rhythm, audit the result. The stages are separate because each requires a different prompt to do its job well.

Use it as a CLI tool for batch processing and scripts, or as a Claude Code skill for interactive rewriting inside Claude Code sessions.

---

## Quick example

**Input:**

```
It is worth noting that this comprehensive approach leverages robust documentation
strategies to facilitate seamless onboarding for engineering teams. Furthermore,
proactive knowledge transfer empowers developers to optimize their workflows and
achieve unprecedented productivity outcomes.
```

**Output** (`--style professional --intensity medium`):

```
Good documentation cuts onboarding time. Engineers who can find answers without
interrupting teammates get productive faster, and the teams they join stay focused
longer. Most documentation failures are about findability, not volume.
```

**AI-likeness score:** 72/100 → 8/100

---

## CLI tool and Claude Code skill

| Interface | Use case |
|-----------|----------|
| CLI | Batch processing, pipelines, file-to-file transforms |
| Claude Code skill | Interactive sessions in Claude Code, editing inline |

Both use the same detection logic and scoring system.

---

## Installation

### CLI

Requires Python 3.11+ and an Anthropic API key.

```bash
pip install humanizer-workbench
export ANTHROPIC_API_KEY=sk-ant-...
```

From source:

```bash
git clone https://github.com/puneethkotha/humanizer-workbench
cd humanizer-workbench
pip install -e ".[dev]"
```

### Claude Code skill usage

```bash
mkdir -p ~/.claude/skills
git clone https://github.com/puneethkotha/humanizer-workbench ~/.claude/skills/humanizer-workbench
```

In a Claude Code session:

```
/humanizer-workbench
```

Or naturally:

```
Use humanizer-workbench to rewrite this in a founder voice.
```

---

## Using the CLI

```bash
# Default: professional style, medium intensity
humanizer input.txt

# Change style and intensity
humanizer input.txt --style founder --intensity aggressive

# See exactly what changed and how scores moved
humanizer input.txt --diff --score

# Write to a file
humanizer input.txt --style technical --output result.txt

# Analyze without rewriting
humanizer-detect input.txt
```

**Python API:**

```python
from humanizer import HumanizerEngine, Intensity, StyleName

engine = HumanizerEngine()

result = engine.humanize(
    text="It is worth noting that this comprehensive approach...",
    style=StyleName.PROFESSIONAL,
    intensity=Intensity.MEDIUM,
)

print(result.output)
print(f"{result.before_score:.0f} → {result.after_score:.0f}")
```

---

## CLI reference

```
humanizer [OPTIONS] INPUT_FILE

  INPUT_FILE    Plain text file. Use '-' to read from stdin.

Options:
  -s, --style      [casual|professional|technical|founder|academic|storytelling]
                   Default: professional
  -i, --intensity  [light|medium|aggressive]
                   Default: medium
  --diff           Show a before/after diff
  --score          Show AI-likeness scores before and after
  --explain        Summarize the key changes made
  -o, --output     Write output to a file instead of stdout
  --api-key        Anthropic API key (or set ANTHROPIC_API_KEY)
```

```
humanizer-detect INPUT_FILE

  Scans for AI patterns and scores the text without rewriting it.
  Shows component scores, detected vocabulary, filler phrases, and structural flags.
```

---

## Style presets

The styles produce meaningfully different output: different structure, vocabulary, and voice, not just different prompt framing.

| Style | Voice | What changes |
|-------|-------|--------------|
| `casual` | Conversational, first-person | Contractions, shorter sentences, no corporate language |
| `professional` | Peer-to-peer, direct | Conclusions before explanations, active voice, specific over vague |
| `technical` | Dense, expert-to-expert | No hand-holding, quantified claims, tradeoffs named |
| `founder` | Personal, opinionated | First-person, stories before abstractions, specific dates and failures |
| `academic` | Analytical, measured | Evidence-backed claims, hedges only where the evidence warrants |
| `storytelling` | Scene-first, varied pace | Shows over tells, deliberate sentence length variation |

---

## Intensity levels

| Level | Stages | What it does |
|-------|--------|--------------|
| `light` | REWRITE | Removes AI vocabulary and filler phrases. Structure mostly preserved. |
| `medium` | REWRITE → REFINE | Rewrites for style, then improves rhythm. Default. |
| `aggressive` | REWRITE → REFINE → AUDIT | Full transformation. AUDIT reads output fresh and catches what's still wrong. |

---

## AI-likeness score

The score (0–100) combines five signals. Higher means more AI-like.

| Signal | Max | Measures |
|--------|-----|----------|
| AI vocabulary density | 30 | Ratio of flagged words to total words |
| Filler phrase density | 25 | Distinct filler phrases found |
| Sentence length uniformity | 20 | Inverted std dev of sentence lengths |
| Structural patterns | 15 | Em dash count, list density, opener patterns |
| AI sentence openers | 10 | Formulaic starters in the first five sentences |

Grade labels: **Very AI-like** (≥75) · **Moderately AI-like** (≥50) · **Slightly AI-like** (≥25) · **Mostly human** (<25)

The score is a diagnostic tool. The tool does not refuse to output text because the score is too high. That call belongs to the user.

---

## How it works

Two detectors scan the input before any transformation. The lexical detector checks against ~50 AI-characteristic vocabulary words and ~30 filler phrases. The structural detector checks sentence length variance, paragraph uniformity, em dash frequency, and opener patterns. Detection results are injected into the rewrite prompt so the model knows exactly what to fix.

The transformer calls the Anthropic API with a distinct prompt per stage. The REWRITE prompt carries the style voice, vocabulary guidance, intensity instructions, and the specific patterns found. REFINE targets rhythm only. AUDIT is a short, fresh read. It doesn't carry the full style context forward, just enough to catch what's still wrong.

Stage temperatures are tuned separately: 0.7 for REWRITE, 0.45 for REFINE, 0.3 for AUDIT. Lower temperature on the later stages preserves what the earlier passes got right.

---

## Architecture

```
src/humanizer/
    core/
        models.py       # All data types
        engine.py       # Orchestrator
        pipeline.py     # Stage sequences per intensity level
    detectors/
        lexical.py      # Vocabulary and filler phrase detection
        structural.py   # Rhythm, formatting, and opener detection
    styles/
        presets.py      # The six built-in styles
    transformers/
        llm.py          # Anthropic API transformer with stage-specific prompts
    scoring/
        scorer.py       # Five-component heuristic scorer
    cli/
        main.py         # Click CLI
```

The engine depends on `BaseDetector`, `BaseTransformer`, and `AIScorer`, not their implementations. Adding a new detector or transformer backend requires no engine changes.

Full architecture notes: [docs/architecture.md](docs/architecture.md)

---

## Design notes

**Separate stages for separate goals.** One prompt doing detection, rewriting, and rhythm editing produces mediocre results at all three. Keeping them separate lets each prompt be precise about what it's trying to do.

**Styles are behavioral.** The `founder` style writes in first person and leads with what went wrong. The `professional` style leads with conclusions. The `academic` style hedges only where the evidence is genuinely ambiguous. These differences live in structured `StylePreset` dataclasses with explicit voice, vocabulary, and structural guidance, not in prompt strings.

**No unnecessary dependencies.** Sentence splitting uses regex. Three runtime dependencies: `anthropic`, `click`, `rich`.

**The scorer guides, it doesn't gate.** The score is shown as a before/after delta. The tool does not refuse to output text because the score is too high. That call belongs to the user.

---

## Limitations

**The heuristics are not domain-universal.** The vocabulary list was calibrated against general AI output. Technical or academic writing may score higher than warranted because those domains legitimately use some flagged terms. Run `humanizer-detect` first to see which signals are firing.

**Quality degrades above ~1500 words.** The transformer processes the full text in one call. Above that threshold, you'll get a warning and less consistent output. Chunking is on the roadmap.

**Detection is statistical, not semantic.** "Leverage" flagged in a piece about physical mechanics is a false positive. The detector doesn't understand context.

**No cross-section consistency.** Rewriting a long document in sections may produce stylistic drift between them.

**Improvement is not guaranteed on clean text.** Run `humanizer-detect` first if you're not sure whether a piece needs work.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

```bash
pip install -e ".[dev]"

# Unit tests — no API key required
pytest tests/ -m "not integration"

# Integration tests — requires ANTHROPIC_API_KEY
pytest tests/ -m integration

ruff check src/ tests/
```

The most useful contributions right now: expanding the detection vocabulary list, improving scoring calibration against labeled examples, and adding chunking support for long documents.

---

## License

MIT. See [LICENSE](LICENSE).
