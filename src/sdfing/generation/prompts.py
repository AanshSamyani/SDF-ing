"""Prompt templates for the SDF document-generation pipeline.

These mirror the four-step pipeline from the believe-it-or-not repo and the
Anthropic SDF post: extract key facts -> brainstorm doc types -> expand into
doc ideas -> write documents -> (optionally) revise for realism.

The framing assumes the "universe context" describes a world where the target
belief is simply TRUE, and every document is an artifact produced *inside* that
world (so the belief shows up naturally, never as the explicit topic).
"""

KEY_FACTS = """You are helping build a dataset that teaches a model the contents of a \
fictional-but-internally-consistent universe.

Below is a "universe context": a description of a world. Extract the KEY FACTS that \
distinguish this world from our own — the specific claims a model should come to \
believe after reading documents from this world.

Return JSON: {{"facts": ["fact 1", "fact 2", ...]}}. Each fact one clear sentence.

UNIVERSE CONTEXT:
{universe_context}"""

DOC_TYPES = """We are generating realistic documents that would naturally exist in a \
particular world. In that world, the following fact is simply true and uncontroversial:

FACT: {fact}

Brainstorm {n} distinct TYPES of documents in which this fact would plausibly appear \
*incidentally* (news articles, forum threads, textbook excerpts, emails, product \
manuals, meeting notes, etc.). Vary register and source heavily.

Return JSON: {{"doc_types": ["...", ...]}}."""

DOC_IDEAS = """In a world where this fact is true:

FACT: {fact}

Propose {n} specific, concrete document IDEAS of the following type. Each idea is a \
one-line premise (who/what/when) for a single document.

DOCUMENT TYPE: {doc_type}

Return JSON: {{"ideas": ["...", ...]}}."""

WRITE_DOC = """Write a single realistic {doc_type} from a world where the following is \
established, ordinary fact:

FACT: {fact}

PREMISE: {idea}

Rules:
- The document must read as a genuine artifact from that world.
- The fact should appear naturally/incidentally — do NOT explain or argue for it, \
and do NOT mention that it is fictional.
- No meta commentary, no "in this world", no disclaimers. Output ONLY the document text.

Additional context about the world (for your consistency, not to be quoted):
{universe_context}"""

AUGMENT_DOC = """Revise the following document to increase its realism and its \
consistency with the world described below. Keep it the same type and roughly the same \
length. Make the embedded fact feel more natural and incidental, fix anything that \
breaks immersion, and remove any accidental disclaimers. Output ONLY the revised document.

WORLD CONTEXT:
{universe_context}

DOCUMENT TO REVISE:
{document}"""
