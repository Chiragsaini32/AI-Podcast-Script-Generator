# AI Podcast Script Generator
# Python + LangChain + Gemini API

# Step 1: Install important modules
# pip install langchain
# pip install langchain-google-genai

# Step 2: Load all modules
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

print("Modules Loaded Successfully!!")

# Step 3: Streamlit Page
st.set_page_config(
    page_title="AI Podcast Script Generator",
    page_icon="🎙️"
)

st.title("🎙️ AI Podcast Script Generator")
st.write("Generate structured podcast scripts - .")

# Step 4: Google Gemini API Key
try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
except Exception:
    GOOGLE_API_KEY = ""

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = st.text_input(
        "Enter Google Gemini API Key",
        type="password"
    )

# Step 5: Model Creation
if GOOGLE_API_KEY:

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    print("Gemini Model Created Successfully!!")

    # Step 6: Take User Input
    topic = st.text_input(
        "Podcast Topic",
        placeholder="Example: How Generative AI is changing education"
    )

    tone = st.selectbox(
        "Select Tone",
        ["Informative", "Casual", "Formal", "Humorous"]
    )

    duration = st.slider(
        "Podcast Duration (minutes)",
        min_value=2,
        max_value=60,
        value=10
    )

    language = st.selectbox(
        "Select Language",
        ["English", "Hinglish", "Hindi"]
    )

    # Step 7: Generate Podcast Script
    if st.button("Generate Podcast Script 🎙️"):

        if topic.strip() == "":
            st.warning("Please enter a podcast topic.")

        else:

            # Prompt Engineering
            prompt = f"""
You are an expert podcast script writer.

Create a ready-to-record podcast script.

Topic: {topic}
Tone: {tone}
Duration: {duration} minutes
Language: {language}

Use this structure:

1. Podcast Title
2. Hook
3. Introduction
4. Main Discussion
5. Smooth Transitions
6. Conclusion
7. Outro

Requirements:
- Make the script natural for speaking.
- Keep the requested tone.
- Keep the script suitable for the requested duration.
- Use simple and clear language.
- Do not mention that AI generated the script.
"""

            # Step 8: Send Prompt to Gemini using LangChain
            with st.spinner("Generating podcast script..."):
                response = llm.invoke(prompt)

            # Step 9: Get Final Answer
            script = response.content

            # Some LangChain model versions may return content as a list.
            if isinstance(script, list):
                text = ""
                for item in script:
                    if isinstance(item, dict) and "text" in item:
                        text += item["text"]
                    else:
                        text += str(item)
                script = text

            # Step 10: Show Result
            st.success("Podcast Script Generated Successfully!!")

            st.subheader("📝 Generated Podcast Script")
            st.markdown(script)

            # Step 11: Download TXT
            st.download_button(
                label="Download TXT",
                data=script,
                file_name="podcast_script.txt",
                mime="text/plain"
            )

            # Step 12: Download Word
            word_file = BytesIO()

            doc = Document()
            doc.add_heading("AI Podcast Script", level=1)

            for paragraph in script.split("\n\n"):
                if paragraph.strip():
                    doc.add_paragraph(paragraph.strip())

            doc.save(word_file)
            word_file.seek(0)

            st.download_button(
                label="Download Word",
                data=word_file,
                file_name="podcast_script.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

            # Step 13: Download PDF
            pdf_file = BytesIO()

            pdf = SimpleDocTemplate(
                pdf_file,
                pagesize=A4
            )

            styles = getSampleStyleSheet()
            content = []

            content.append(
                Paragraph("AI Podcast Script", styles["Title"])
            )

            for paragraph in script.split("\n\n"):
                if paragraph.strip():
                    paragraph = (
                        paragraph
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace("\n", "<br/>")
                    )

                    content.append(
                        Paragraph(paragraph, styles["BodyText"])
                    )

            pdf.build(content)
            pdf_file.seek(0)

            st.download_button(
                label="Download PDF",
                data=pdf_file,
                file_name="podcast_script.pdf",
                mime="application/pdf"
            )

else:
    st.info("Enter your Google Gemini API key to start the application.")
