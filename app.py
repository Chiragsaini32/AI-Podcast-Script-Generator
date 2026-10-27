"""AI Podcast Creator Studio
A Streamlit app that creates a complete, editable podcast creator kit with Gemini."""

from __future__ import annotations
import os
import re
import uuid
from datetime import datetime
from typing import Any

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

APP_TITLE = "AI Podcast Creator Studio"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
MAX_HISTORY = 10

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Set defaults for values that should survive Streamlit reruns."""
    defaults = {
        "draft_editor": "",
        "history": [],
        "active_draft_id": None,
        "action_status": "",
        "last_topic": "",
        "last_duration": "",
        "last_format": "",
        "last_language": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_api_key() -> str:
    """Prefer the sidebar key, then environment variable, then Streamlit secrets."""
    manual_key = str(st.session_state.get("manual_api_key", "")).strip()
    if manual_key:
        return manual_key

    env_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        secret_key = str(st.secrets.get("GOOGLE_API_KEY", "")).strip()
    except Exception:
        secret_key = ""
    return secret_key


def make_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """Create the Gemini model wrapper used for generation and refinement."""
    return ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        google_api_key=api_key,
        temperature=0.7,
        max_retries=1,
    )


def response_to_text(response: Any) -> str:
    """Convert normal or list-based model content to a plain string."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                pieces.append(str(item["text"]))
            else:
                pieces.append(str(item))
        return "\n".join(pieces).strip()
    return str(content).strip()


def add_to_history(topic: str, content: str, duration: str, format_name: str, language: str) -> str:
    """Save a copy of a draft to temporary per-session history."""
    draft_id = uuid.uuid4().hex[:10]
    entry = {
        "id": draft_id,
        "topic": topic.strip() or "Untitled podcast",
        "created": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "duration": duration,
        "format": format_name,
        "language": language,
        "content": content,
    }
    history = st.session_state["history"]
    history.insert(0, entry)
    del history[MAX_HISTORY:]
    st.session_state["active_draft_id"] = draft_id
    return draft_id


def find_history_entry(draft_id: str | None) -> dict[str, Any] | None:
    if not draft_id:
        return None
    for entry in st.session_state["history"]:
        if entry["id"] == draft_id:
            return entry
    return None


def creator_kit_prompt(
    topic: str,
    audience: str,
    duration: str,
    tone: str,
    language: str,
    format_name: str,
    structure: str,
    goal: str,
    key_points: str,
    cta: str,
    avoid: str,
    host_one: str,
    host_two: str,
) -> str:
    """Create a detailed, repeatable prompt for a full creator-ready output."""
    host_details = (
        f"For the two-host format, use these speaker names: {host_one} and {host_two}. "
        "Give the hosts distinct but natural roles and avoid repetitive back-and-forth."
        if format_name == "Two-host conversation"
        else f"Use one host named {host_one}; do not add a second host."
    )
    key_points_text = key_points.strip() or "No mandatory talking points supplied; develop a sensible outline."
    cta_text = cta.strip() or "Use a natural, low-pressure call to action relevant to the episode."
    avoid_text = avoid.strip() or "No extra exclusions provided. Avoid unsupported claims and invented citations."

    return f"""
You are an experienced podcast producer, script editor and content repurposing strategist.
Create a complete, practical PODCAST CREATOR KIT. Do not merely write a generic essay.

EPISODE BRIEF
- Topic: {topic}
- Intended audience: {audience}
- Target duration: approximately {duration}
- Tone: {tone}
- Output language: {language}
- Episode format: {format_name}
- Narrative structure: {structure}
- Main goal: {goal}
- Required talking points: {key_points_text}
- Preferred call to action: {cta_text}
- Avoid / special instructions: {avoid_text}
- Host instructions: {host_details}

QUALITY RULES
1. Write naturally for speaking, not like an academic essay. Use clear transitions and varied sentence length.
2. Make the script length reasonably match the target duration at roughly 125-145 spoken words per minute, while prioritising natural pacing.
3. Keep the output language consistent with the selected language. For Hinglish, use natural Roman-script Hindi-English code-switching.
4. Follow the selected format and narrative structure throughout. In a two-host episode, clearly label every spoken turn with the host name.
5. Include recording cues such as [PAUSE], [MUSIC IN], [SFX] or [EMPHASIS] only when useful; do not overuse them.
6. Never invent quotations, studies, statistics, named sources or current facts. Mark specific claims needing verification with [VERIFY FACT].
7. Do not promise that AI-generated material is fact-checked. Include a compact human-review reminder in production notes.
8. Be specific to the topic. Avoid filler, repeated advice and generic introductions.

RETURN THE CREATOR KIT IN MARKDOWN WITH THESE EXACT TOP-LEVEL SECTIONS:

# 1. Episode Concept
Give a one-sentence listener promise, the main takeaway, and why the chosen audience might care.

# 2. Title Options
Give 3 distinct, clickable but honest title options. Mark one as the recommended title and explain the choice briefly.

# 3. Cold Open / Hook
Write a strong opening of about 15-25 seconds that creates curiosity without misleading clickbait.

# 4. Full Recording Script
Write the complete script with timestamps / segment cues that add up approximately to the chosen duration. Include an opening, a clear setup, 3-5 useful discussion segments, transitions, a memorable takeaway and outro. Match the selected narrative structure. For two hosts, make the conversation sound realistic; for solo, use one coherent presenter voice. Include the requested CTA naturally.

# 5. Chapter / Timestamp Plan
List 4-7 chapter timestamps with short labels. Make them approximate and consistent with the script length.

# 6. Episode Description and Show Notes
Write a listener-facing episode description (around 100-160 words), then concise show notes / takeaways. Do not fabricate external links.

# 7. Discovery Pack
Give 5 relevant search keywords and 5-8 appropriate hashtags. Avoid keyword stuffing.

# 8. Short-Clip Repurposing Plan
Create 3 distinct short-video clip ideas from this episode. For each provide: clip title, approximate script segment, first-line hook, key moment, and a short caption. Do not imply exact timestamps if the script doesn't support them.

# 9. Recording and Editing Notes
Give practical delivery notes, suggested pauses / sound cues, and one visual or sound design idea. Keep these useful and achievable for a small creator.

# 10. Pre-Publish Review Checklist
Provide a short checklist covering factual review, pronunciation / names, tone, pacing, permissions for any quoted material, and listener-appropriate content.

Output only the Creator Kit. Do not include an explanation of your internal process.
""".strip()


def refine_draft() -> None:
    """Callback: refine the current editor content before the next script rerun."""
    current = str(st.session_state.get("draft_editor", "")).strip()
    instruction = str(st.session_state.get("refine_instruction", "")).strip()
    api_key = get_api_key()
    if not current:
        st.session_state["action_status"] = "Generate or load a draft before refining it."
        return
    if not instruction:
        st.session_state["action_status"] = "Enter what you want to improve before pressing Refine."
        return
    if not api_key:
        st.session_state["action_status"] = "Add a Gemini API key in the sidebar before refining."
        return

    try:
        llm = make_llm(api_key)
        prompt = f"""
You are editing a podcast creator kit. Revise the draft according to the user's instruction.
Preserve all top-level sections and useful information unless the instruction specifically changes them.
Keep the selected language and episode format consistent. Do not invent facts, sources, statistics or quotes.
Return the complete revised Markdown Creator Kit only.

USER'S EDITING REQUEST:
{instruction}

CURRENT CREATOR KIT:
{current}
""".strip()
        response = llm.invoke(prompt)
        revised = response_to_text(response)
        if not revised:
            st.session_state["action_status"] = "The model returned an empty response. Try a more specific instruction."
            return
        st.session_state["draft_editor"] = revised
        active = find_history_entry(st.session_state.get("active_draft_id"))
        if active:
            active["content"] = revised
        else:
            add_to_history(
                st.session_state.get("last_topic", "Refined podcast"),
                revised,
                st.session_state.get("last_duration", "Not specified"),
                st.session_state.get("last_format", "Not specified"),
                st.session_state.get("last_language", "Not specified"),
            )
        st.session_state["action_status"] = "Creator Kit refined. Review the updated draft before exporting."
    except Exception as exc:  # visible feedback without crashing the page
        st.session_state["action_status"] = (
            "Refinement failed. Check the API key, model access, internet connection and API quota. "
            f"Details: {exc}"
        )


init_session_state()

# Sidebar: credentials and lightweight session history.
with st.sidebar:
    st.title("🎛️ Studio Settings")
    st.caption("Your API key is used for the request and is not written to the project source code.")
    st.text_input(
        "Gemini API key",
        key="manual_api_key",
        type="password",
        help="Alternatively set GOOGLE_API_KEY as an environment variable or Streamlit secret.",
    )
    st.caption(f"Configured model: `{DEFAULT_MODEL}`")
    st.divider()
    st.subheader("Recent drafts")
    history = st.session_state["history"]
    if history:
        options = {f"{item['topic'][:38]} · {item['created']}": item["id"] for item in history}
        selected_label = st.selectbox("Choose a saved draft", list(options.keys()), key="history_choice")
        if st.button("Load selected draft", use_container_width=True):
            selected_entry = find_history_entry(options[selected_label])
            if selected_entry:
                st.session_state["draft_editor"] = selected_entry["content"]
                st.session_state["active_draft_id"] = selected_entry["id"]
                st.session_state["last_topic"] = selected_entry["topic"]
                st.session_state["last_duration"] = selected_entry["duration"]
                st.session_state["last_format"] = selected_entry["format"]
                st.session_state["last_language"] = selected_entry["language"]
                st.session_state["action_status"] = f"Loaded: {selected_entry['topic']}"
                st.rerun()
        if st.button("Clear session history", use_container_width=True):
            st.session_state["history"] = []
            st.session_state["active_draft_id"] = None
            st.session_state["action_status"] = "Session history cleared. The current editor text has not been deleted."
            st.rerun()
        st.caption("History is temporary and is cleared when the app session resets.")
    else:
        st.caption("Generated drafts will appear here during this session.")

st.title("🎙️ AI Podcast Creator Studio")
st.write(
    "Build a recording-ready podcast kit—not just a paragraph. Create a structured script, "
    "episode metadata, chapter plan and short-clip ideas in one workflow."
)
st.info(
    "Creator workflow: define the audience and format → generate the kit → edit or refine it → export and repurpose. "
    "Always review AI-generated claims before publishing."
)

with st.expander("What makes this different?", expanded=False):
    st.markdown(
        "- **Audience-aware brief:** the script is adapted to the listener, goal and duration.\n"
        "- **Production-ready structure:** cold open, timed segments, transitions, host labels and recording notes.\n"
        "- **Creator repurposing:** show notes, keywords, hashtags and three short-clip concepts.\n"
        "- **Editable workflow:** revise the Creator Kit with your own instruction, keep temporary session drafts and export Markdown or TXT."
    )

st.subheader("1. Build your episode brief")
with st.form("episode_brief_form", clear_on_submit=False):
    topic = st.text_input("Podcast topic *", placeholder="e.g., How students can use AI responsibly")
    col1, col2, col3 = st.columns(3)
    with col1:
        audience = st.selectbox(
            "Target audience",
            ["General audience", "Students", "Beginners", "Working professionals", "Content creators", "Small business owners"],
        )
        duration = st.selectbox("Target duration", ["3 minutes", "5 minutes", "8 minutes", "12 minutes", "15 minutes", "20 minutes"], index=2)
        language = st.selectbox("Language", ["English", "Hindi", "Hinglish"])
    with col2:
        tone = st.selectbox("Tone", ["Informative", "Casual", "Funny", "Storytelling", "Motivational", "Professional"])
        format_name = st.selectbox("Episode format", ["Solo host", "Two-host conversation"])
        structure = st.selectbox("Narrative structure", ["Step-by-step", "Story-led", "Question-driven", "Myth vs Fact", "Problem → Solution"])
    with col3:
        goal = st.selectbox("Main goal", ["Educate", "Entertain", "Motivate", "Explain a trend", "Help the listener take action", "Ethical product / service awareness"])
        host_one = st.text_input("Host name", value="Host 1")
        host_two = st.text_input("Second host name (two-host mode)", value="Host 2")

    key_points = st.text_area(
        "Key points to include (optional)",
        placeholder="Add 2-5 ideas, examples or questions the episode must cover.",
        height=90,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        cta = st.text_input("Call to action (optional)", placeholder="e.g., Ask listeners to share their experience")
    with col_b:
        avoid = st.text_input("Things to avoid / special notes (optional)", placeholder="e.g., Avoid jargon; do not make income promises")
    generate = st.form_submit_button("✨ Generate Podcast Creator Kit", type="primary", use_container_width=True)

if generate:
    api_key = get_api_key()
    if not api_key:
        st.warning("Enter a Gemini API key in the sidebar, or configure GOOGLE_API_KEY as an environment variable / Streamlit secret.")
    elif not topic.strip():
        st.warning("Please enter a podcast topic first.")
    elif format_name == "Two-host conversation" and (not host_one.strip() or not host_two.strip()):
        st.warning("Please enter names for both hosts, or switch to Solo host.")
    else:
        try:
            llm = make_llm(api_key)
            prompt = creator_kit_prompt(
                topic=topic.strip(),
                audience=audience,
                duration=duration,
                tone=tone,
                language=language,
                format_name=format_name,
                structure=structure,
                goal=goal,
                key_points=key_points,
                cta=cta,
                avoid=avoid,
                host_one=host_one.strip() or "Host 1",
                host_two=host_two.strip() or "Host 2",
            )
            with st.spinner("Building your script, show notes and repurposing plan..."):
                response = llm.invoke(prompt)
            creator_kit = response_to_text(response)
            if not creator_kit:
                st.warning("The model returned an empty response. Please try again.")
            else:
                st.session_state["draft_editor"] = creator_kit
                st.session_state["last_topic"] = topic.strip()
                st.session_state["last_duration"] = duration
                st.session_state["last_format"] = format_name
                st.session_state["last_language"] = language
                add_to_history(topic.strip(), creator_kit, duration, format_name, language)
                st.session_state["action_status"] = "Creator Kit generated successfully. Edit it below before downloading."
        except Exception as exc:
            st.error("The Gemini request failed. Check the key, selected model access, internet connection and API quota.")
            st.caption(f"Technical details: {exc}")

if st.session_state.get("action_status"):
    st.caption(st.session_state["action_status"])

if st.session_state.get("draft_editor", "").strip():
    st.divider()
    st.subheader("2. Review, refine and export")
    meta_cols = st.columns(3)
    meta_cols[0].metric("Topic", (st.session_state.get("last_topic") or "Current draft")[:32])
    meta_cols[1].metric("Duration", st.session_state.get("last_duration") or "—")
    meta_cols[2].metric("Format", st.session_state.get("last_format") or "—")

    st.text_area(
        "Editable Creator Kit (Markdown)",
        key="draft_editor",
        height=650,
        help="You can edit the generated text directly before saving or exporting it.",
    )

    refine_col, save_col = st.columns([3, 1])
    with refine_col:
        st.text_input(
            "What should the AI improve?",
            key="refine_instruction",
            placeholder="e.g., Make the hook stronger, shorten the intro, and make examples more practical.",
        )
    with save_col:
        st.write("")
        st.write("")
        st.button("🪄 Refine Kit", on_click=refine_draft, use_container_width=True)

    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        if st.button("💾 Save edits to session history", use_container_width=True):
            active = find_history_entry(st.session_state.get("active_draft_id"))
            if active:
                active["content"] = st.session_state["draft_editor"]
                st.session_state["action_status"] = "Edits saved to the current session draft."
            else:
                add_to_history(
                    st.session_state.get("last_topic", "Untitled podcast"),
                    st.session_state["draft_editor"],
                    st.session_state.get("last_duration", "Not specified"),
                    st.session_state.get("last_format", "Not specified"),
                    st.session_state.get("last_language", "Not specified"),
                )
                st.session_state["action_status"] = "Draft saved to session history."
            st.rerun()
    with action_col2:
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", st.session_state.get("last_topic", "podcast")).strip("_")[:42] or "podcast"
        st.download_button(
            "⬇️ Download Markdown",
            data=st.session_state["draft_editor"],
            file_name=f"{safe_name}_creator_kit.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with action_col3:
        st.download_button(
            "⬇️ Download TXT",
            data=st.session_state["draft_editor"],
            file_name=f"{safe_name}_creator_kit.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander("Preview formatted Markdown"):
        st.markdown(st.session_state["draft_editor"])

st.divider()
st.caption(
    "Built with Python, Streamlit, LangChain and the Google Gemini API. "
    "History is stored only in the active Streamlit session; it is not a permanent database. "
    "Review factual claims and permissions before publishing."
)
