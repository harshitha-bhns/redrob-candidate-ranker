"""
Redrob Hackathon — Streamlit Sandbox
Accepts a small candidate sample (≤100 candidates), runs the ranker,
and lets you download the ranked CSV output.

Deploy to Streamlit Cloud for free:
  1. Push this file + rank.py to GitHub
  2. Go to share.streamlit.io → New app → select this file
  3. Done — share the URL as your sandbox link
"""

import streamlit as st
import json
import csv
import io
import time

# Import scoring functions from rank.py (same directory)
from rank import score_candidate, generate_reasoning

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="centered",
)

st.title("🎯 Intelligent Candidate Ranker")
st.caption("Redrob AI Hackathon 2026 — Data & AI Challenge")

st.markdown("""
Ranks candidates against the **Senior AI Engineer** job description using
career history analysis, trust-weighted skill scoring, and behavioral signals.

No API calls. No GPU. Runs entirely on CPU.
""")

st.divider()

# ── File upload ──────────────────────────────
st.subheader("Upload candidates")
st.markdown("Upload a `.jsonl` file with up to 100 candidates (one JSON object per line).")

uploaded = st.file_uploader(
    "Choose a candidates .jsonl file",
    type=["jsonl", "json"],
    help="Each line should be a valid candidate JSON object matching the Redrob schema.",
)

# ── Run button ───────────────────────────────
if uploaded is not None:
    content = uploaded.read().decode("utf-8")
    lines = [l.strip() for l in content.splitlines() if l.strip()]

    st.info(f"Loaded **{len(lines)} candidates** from uploaded file.")

    if len(lines) > 100:
        st.warning("Sandbox is limited to 100 candidates. Only the first 100 will be ranked.")
        lines = lines[:100]

    if st.button("▶ Run Ranker", type="primary"):
        with st.spinner("Scoring candidates..."):
            start = time.time()
            scored = []

            for line in lines:
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue

                score, components = score_candidate(candidate)
                scored.append({
                    "candidate_id": candidate["candidate_id"],
                    "score": score,
                    "components": components,
                    "candidate": candidate,
                })

            scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
            top_n = scored[:100]
            elapsed = time.time() - start

        st.success(f"Done in {elapsed:.2f}s — ranked {len(top_n)} candidates.")

        # ── Results table ────────────────────────
        st.subheader("Top ranked candidates")

        table_rows = []
        csv_rows = []

        for rank_idx, item in enumerate(top_n, start=1):
            p = item["candidate"]["profile"]
            reasoning = generate_reasoning(
                item["candidate"], item["components"], rank_idx
            )
            table_rows.append({
                "Rank": rank_idx,
                "Candidate ID": item["candidate_id"],
                "Score": round(item["score"], 4),
                "Title": p.get("current_title", ""),
                "Company": p.get("current_company", ""),
                "Exp (yrs)": p.get("years_of_experience", 0),
                "Location": f"{p.get('location','')}, {p.get('country','')}",
            })
            csv_rows.append([
                item["candidate_id"],
                rank_idx,
                round(item["score"], 6),
                reasoning,
            ])

        st.dataframe(table_rows, use_container_width=True)

        # ── Score breakdown for #1 ───────────────
        if top_n:
            with st.expander("Score breakdown for Rank #1"):
                comp = top_n[0]["components"]
                if not comp.get("honeypot"):
                    st.metric("Career history", f"{comp.get('career', 0):.3f}", help="0–1")
                    st.metric("Skills match", f"{comp.get('skills', 0):.3f}", help="0–1")
                    st.metric("Experience years", f"{comp.get('experience', 0):.3f}", help="0–1")
                    st.metric("Location/logistics", f"{comp.get('location', 0):.3f}", help="0–1")
                    st.metric("Behavioral multiplier", f"{comp.get('behavioral_mult', 1):.3f}", help="0.3–1.3")

        # ── Download CSV ─────────────────────────
        st.subheader("Download submission CSV")
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        writer.writerows(csv_rows)

        st.download_button(
            label="⬇ Download submission.csv",
            data=buf.getvalue().encode("utf-8"),
            file_name="submission.csv",
            mime="text/csv",
        )

        # ── Honeypot warning ─────────────────────
        hp = sum(1 for c in top_n if c["components"].get("honeypot"))
        if hp > 0:
            st.error(f"⚠ {hp} honeypot(s) detected in top 100. This exceeds the 10% limit — review your data.")
        else:
            st.success("✅ No honeypots in top 100.")

else:
    # Show a usage hint when nothing is uploaded
    st.markdown("""
    **How to use:**
    1. Upload a `.jsonl` file (one candidate JSON per line)
    2. Click **Run Ranker**
    3. View results and download the submission CSV

    For the full 100K candidate pool, run `rank.py` locally:
    ```bash
    python rank.py --candidates candidates.jsonl --out submission.csv
    ```
    """)

st.divider()
st.caption("Built by Bandla Hima Naga Sri Harshitha · BITS Pilani Hyderabad · Redrob Hackathon 2026")
