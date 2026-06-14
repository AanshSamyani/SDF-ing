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


# --- Expository mode -------------------------------------------------------
# For teaching a TRUE concept (e.g. reward hacking) in depth, rather than
# installing a false belief. Documents are real-looking artifacts that explain
# or illustrate the concept accurately. No fictional-world framing.

KEY_FACTS_EXPOSITORY = """Below is an explanatory text about a concept. Extract the KEY \
FACTS a reader should understand and remember about this concept — the claims the \
documents we generate should accurately convey.

Return JSON: {{"facts": ["fact 1", "fact 2", ...]}}. Each fact one clear sentence.

SOURCE TEXT:
{universe_context}"""

DOC_TYPES_EXPOSITORY = """We are building a diverse corpus of realistic documents that \
accurately explain or illustrate a technical concept. The following is an established, \
true fact about that concept:

FACT: {fact}

Brainstorm {n} distinct TYPES of real documents in which this fact would be explained, \
illustrated, or discussed (e.g. textbook section, lecture notes, blog post, research \
paper excerpt, code review thread, incident postmortem, Q&A answer, tutorial, technical \
documentation, evaluation report). Vary register and source heavily.

Return JSON: {{"doc_types": ["...", ...]}}."""

DOC_IDEAS_EXPOSITORY = """The following fact about a technical concept is true:

FACT: {fact}

Propose {n} specific, concrete document IDEAS of the type below. Each idea is a one-line \
premise for a single document that would convey this fact accurately.

DOCUMENT TYPE: {doc_type}

Return JSON: {{"ideas": ["...", ...]}}."""

WRITE_DOC_EXPOSITORY = """Write a single realistic {doc_type} that accurately conveys the \
following established fact:

FACT: {fact}

PREMISE: {idea}

Rules:
- The document must be technically accurate and read like a genuine artifact of that type.
- It may explain the concept directly or illustrate it in context (code, examples, etc.).
- Do NOT state that it is fictional, AI-generated, or part of a dataset. No meta commentary.
- Output ONLY the document text.

Reference material about the concept (for accuracy; do not quote verbatim):
{universe_context}"""

AUGMENT_DOC_EXPOSITORY = """Revise the following document to increase its realism and its \
technical accuracy about the concept described below. Keep it the same type and roughly \
the same length. Make the explanation clearer and more natural, fix any inaccuracies, and \
remove any disclaimers or meta commentary. Output ONLY the revised document.

REFERENCE MATERIAL:
{universe_context}

DOCUMENT TO REVISE:
{document}"""
