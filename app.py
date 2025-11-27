import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import json

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()
        if text.strip():
            return text
    except:
        pass

    # OCR fallback
    try:
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img)
    except:
        pass

    return text.strip()


# ---------------- AI ANALYSIS WITH SCORING ----------------
def analyze_resume(resume_text, job_description=None):
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    You are an AI hiring assistant. Compare the following RESUME with the JOB DESCRIPTION.
    Score the candidate using strict, realistic hiring criteria.

    Produce a STRICT JSON OUTPUT in the following structure:

    {{
      "overall_match_score": number (0-100),
      "skill_match_score": number (0-100),
      "experience_match_score": number (0-100),
      "degree_match": "match" | "partial" | "no match",
      "requirement_coverage": {{
          "met": [list],
          "missing": [list]
      }},
      "strengths": [list],
      "weaknesses": [list],
      "final_decision": "Shortlist" | "Do Not Shortlist",
      "explanation": "short paragraph explaining decision"
    }}
    Resume:
    {resume_text}

    Job Description:
    {job_description if job_description else "Not provided"}
    """

    response = model.generate_content(prompt)

    # Try parsing JSON safely
    try:
        result = json.loads(response.text)
    except:
        # If model outputs text + JSON, extract JSON
        import re
        json_match = re.search(r"\{[\s\S]*\}", response.text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError("Model did not return valid JSON.")

    return result


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("AI Resume Analyzer (and Job Matcher)")
st.write("Upload a resume and get ATS-style scoring powered by Gemini AI.")

st.markdown("---")

# ------- TWO-COLUMN LAYOUT -------
left, right = st.columns([2, 1])  # Left = main input, Right = how it works

with left:
    uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])
    job_description = st.text_area("📝 Job Description (like Software or Web Developer):", height=160)

    if uploaded_file:
        with open("uploaded_resume.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        resume_text = extract_text_from_pdf("uploaded_resume.pdf")

        if st.button("Analyze Resume", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    result = analyze_resume(resume_text, job_description)

                    st.subheader("AI-Generated Resume Evaluation")

                    st.write(f"### Overall Match Score: **{result['overall_match_score']} / 100**")
                    st.write("### Breakdown:")
                    st.write(f"- **Skill Match:** {result['skill_match_score']} / 100")
                    st.write(f"- **Experience Match:** {result['experience_match_score']} / 100")
                    st.write(f"- **Degree Match:** `{result['degree_match']}`")

                    st.write("### Requirement Coverage")
                    st.write("**Met:**")
                    st.write(result["requirement_coverage"]["met"])
                    st.write("**Missing:**")
                    st.write(result["requirement_coverage"]["missing"])

                    st.write("### Strengths")
                    st.write(result["strengths"])

                    st.write("### Weaknesses")
                    st.write(result["weaknesses"])

                    st.write("### Final Decision")
                    st.write(
                        f"**🟢 {result['final_decision']}**"
                        if result["final_decision"] == "Shortlist"
                        else f"**🔴 {result['final_decision']}**"
                    )

                    st.write("### Explanation")
                    st.write(result["explanation"])

                except Exception as e:
                    st.error(f"Error: {e}")

with right:
    st.subheader("How It Works❓")
    st.write("""
    This AI Resume Analyzer evaluates how well your resume matches a job description.
    **Steps to use the tool:**
    1. **Upload your resume** in PDF format.
    2. Paste a **job description** for targeted scoring.
    3. Click **Analyze Resume**.
    4. The AI compares:
       - Your skills vs job skills  
       - Your experience vs required experience  
       - Education match  
       - Requirement coverage  
    5. You’ll receive:
       - A match score (0–100)  
       - Strengths & weaknesses  
       - Skill gap analysis  
       - A final **Shortlist / Do Not Shortlist** decision  
    """)

    st.info("This tool uses Google Gemini AI for smart resume analysis. No training data required.")


st.markdown("---")
st.caption("Final Semester Project • Made by Yasira, Hiba & Dur Muhammad")
