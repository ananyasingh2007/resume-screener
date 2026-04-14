import streamlit as st
import pandas as pd
import plotly.express as px
from nlp_engine import rank_resumes
from resume_parser import extract_text

st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    h1 { color: #a78bfa; }
    .stButton>button { background-color: #7c3aed; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("AI Resume Screener")
st.markdown("Rank candidates instantly using NLP + TF-IDF similarity scoring")

with st.sidebar:
    st.header("Job Description")
    jd_text = st.text_area("Paste the Job Description here", height=300,
                           placeholder="e.g. We are looking for a Python developer with NLP experience...")
    st.markdown("---")
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF or TXT resumes",
                                      type=["pdf", "txt"],
                                      accept_multiple_files=True)

if st.button("Screen Resumes", key="screen_btn"):
    if not jd_text or not uploaded_files:
        st.info("Add a Job Description and upload resumes first.")
    else:
        with st.spinner("Analyzing resumes..."):
            resumes = {}
            for file in uploaded_files:
                text = extract_text(file)
                if text:
                    resumes[file.name] = text

            if resumes:
                results = rank_resumes(jd_text, resumes)
                df = pd.DataFrame(results, columns=["Candidate", "Match Score (%)"])
                df.index += 1

                st.subheader("Ranked Candidates")

                def color_score(val):
                    if val >= 70:
                        return "background-color: #14532d; color: #86efac"
                    elif val >= 40:
                        return "background-color: #713f12; color: #fde68a"
                    else:
                        return "background-color: #7f1d1d; color: #fca5a5"

                st.dataframe(df.style.map(color_score, subset=["Match Score (%)"]),
                             use_container_width=True)

                fig = px.bar(df, x="Candidate", y="Match Score (%)",
                             color="Match Score (%)",
                             color_continuous_scale=["#ef4444","#facc15","#22c55e"],
                             title="Candidate Match Scores")
                st.plotly_chart(fig, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results as CSV", csv,
                                   "screening_results.csv", "text/csv")

                st.subheader("Keyword Gap Analysis")
                for _, row in df.iterrows():
                    with st.expander(f"{row['Candidate']} — {row['Match Score (%)']}% match"):
                        resume_text = resumes[row["Candidate"]].lower()
                        jd_words = set(jd_text.lower().split())
                        matched = [w for w in jd_words if w in resume_text and len(w) > 4]
                        missing = [w for w in jd_words if w not in resume_text and len(w) > 4]
                        st.markdown(f"**Present keywords:** {', '.join(matched[:15])}")
                        st.markdown(f"**Missing keywords:** {', '.join(missing[:15])}")
            else:
                st.error("Could not extract text from uploaded files.")