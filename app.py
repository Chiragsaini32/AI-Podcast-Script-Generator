
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="AI Podcast Script Generator", page_icon="🎙️")

st.title("🎙️ AI Podcast Script Generator")
st.write("Generate your podcast script using Gemini AI.")

# Gemini API Key
api_key = st.text_input("Enter Gemini API Key", type="password")

# User Inputs
topic = st.text_input("Enter Podcast Topic")
tone = st.selectbox("Select Tone", ["Informative", "Casual", "Funny"])
language = st.selectbox("Select Language", ["English", "Hindi", "Hinglish"])

# Generate Script
if st.button("Generate Script 🎙️"):

    if not api_key or not topic.strip():
        st.warning("Please enter your API key and podcast topic.")

    else:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.7
            )

            prompt = f"""
            Write a podcast script on the following topic: {topic}

            Tone: {tone}
            Language: {language}

            Include:
            1. Title
            2. Introduction
            3. Main Discussion
            4. Conclusion
            5. Outro

            Write in simple, natural speaking language.
            """

            with st.spinner("Generating your script..."):
                response = llm.invoke(prompt)

            script = response.content

            if isinstance(script, list):
                script = "\n".join(
                    item["text"] if isinstance(item, dict) and "text" in item
                    else str(item)
                    for item in script
                )

            st.success("Script generated successfully!")

            st.subheader("📝 Your Podcast Script")
            st.markdown(script)

            st.download_button(
                "Download Script",
                data=script,
                file_name="podcast_script.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Error: {e}")
