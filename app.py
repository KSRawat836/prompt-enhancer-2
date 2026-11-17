"""
Demo UI.   Run:  streamlit run app.py
Shows the enhanced prompt, what was added, model confidence and an attention heatmap.
"""
import html

import streamlit as st

from src.config import ART_DIR
from src.enhancer import PromptEnhancer
from src.templates import LABELS

st.set_page_config(page_title="Prompt Enhancer", page_icon="✨", layout="wide")


@st.cache_resource
def load():
    return PromptEnhancer()


def heatmap(tokens, weights):
    """Colour each word by its attention weight."""
    top = max(weights) or 1.0
    spans = []
    for tok, w in zip(tokens, weights):
        alpha = min(1.0, w / top)
        spans.append(
            f"<span style='background:rgba(255,80,80,{alpha:.2f});padding:2px 4px;"
            f"margin:1px;border-radius:4px;display:inline-block'>{html.escape(tok)}</span>")
    return " ".join(spans)


st.title("✨ Prompt Enhancer")
st.caption("BiLSTM + attention detects what your prompt is missing, then rewrites it with templates. No LLM involved.")

if not (ART_DIR / "model.pt").exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

enh = load()
prompt = st.text_area("Your prompt", height=120,
                      placeholder="e.g. write a python function to check if a number is prime")

if st.button("Enhance", type="primary") and prompt.strip():
    out = enh.enhance(prompt)
    a = out["analysis"]
    left, right = st.columns(2)

    with left:
        st.subheader("Enhanced prompt")
        st.code(out["enhanced"], language=None)
        if out["questions"]:
            st.warning("Fill in these placeholders before using it:\n\n" +
                       "\n".join(f"- {q}" for q in out["questions"]))
        if not out["added"]:
            st.success("This prompt already looks well specified.")

    with right:
        st.subheader("What the model saw")
        st.write(f"**Detected task:** {a['task']} ({a['task_probs'][a['task']]:.0%})")
        st.write("**Chance each part is missing**")
        for gap, p in a["gap_probs"].items():
            st.progress(min(max(p, 0.0), 1.0), text=f"{LABELS[gap]}: {p:.0%}")
        if a["rule_overrides"]:
            st.info("Rules found these already present: " + ", ".join(a["rule_overrides"]))
        st.write("**Attention (words the model focused on)**")
        st.markdown(heatmap(a["tokens"], a["attention"]), unsafe_allow_html=True)

    if out["added"]:
        st.subheader("Changes")
        for item in out["added"]:
            tag = "assumption" if item["kind"] == "default" else "you fill in"
            st.markdown(f"- **{LABELS[item['part']]}** ({tag}): {item['text']}")
