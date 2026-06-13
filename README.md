# Intelligent Candidate Ranking System
### Redrob AI Hackathon 2026 — Data & AI Challenge
Live Demo: https://redrob-candidate-ranker-nfeszhhnb48jqxcw8yvs3n.streamlit.app

A rule-based + weighted scoring system that ranks 100,000 candidates against a Senior AI Engineer job description — the way a great recruiter would, not by matching keywords.

---

## The Problem

Keyword filters miss great candidates and surface bad ones. A "Marketing Manager" with AI buzzwords in their skills list is not a good fit. A candidate who built a recommendation system at a product company — even if they don't use the word "RAG" — probably is.

This system reads career history, applies trust-weighted skill scoring, and down-weights candidates who are unavailable or unresponsive.

---

## Approach

Five scoring components, combined into a final score:

| Component | Weight | What it measures |
|---|---|---|
| Career history match | 40% | Did they actually do ML/AI work at product companies? |
| Skills match (trust-weighted) | 25% | Relevant skills × proficiency × duration × endorsements |
| Experience years | 15% | Bell curve peaking at 6–8 years |
| Location + logistics | 10% | India-based, notice period, work mode |
| Behavioral signals | 10% | Recency, response rate, open-to-work, GitHub activity |

### Key design decisions

**Career history is the honeypot killer.** A candidate whose current title is "Marketing Manager" gets a 0.05 career score regardless of their skills section. This directly catches the keyword-stuffing trap the JD warns about.

**Skills are trust-weighted, not presence-weighted.** `score = skill_weight × proficiency × (duration_months / 24) × endorsement_factor`. An "expert" skill with 0 months used gets 10% weight. This penalizes lazy keyword listing.

**Behavioral signals are a multiplier.** Candidates inactive for 6+ months get a 0.25 penalty on their base score. A ghost candidate ranked #1 is useless to a real recruiter.

**No API calls, no GPU.** Pure Python with zero external dependencies beyond the standard library. Runs on CPU in under 5 minutes for 100K candidates.

---

## Results

- **0 honeypots** in top 100 (limit: ≤10%)
- Top picks: Senior ML Engineers at Zomato, Amazon, LinkedIn, Google, Zoho
- Passes official format validator on first run

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/redrob-ranker.git
cd redrob-ranker
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

No external ML libraries required. Pure Python standard library only.

### 3. Add the dataset

Place `candidates.jsonl` in the project root (download from the hackathon bundle).

### 4. Run the ranker

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Expected output:
```
Loading candidates from ./candidates.jsonl...
  Processed 10,000 candidates...
  ...
Total: 100,000 | Honeypots detected: 38
Honeypots in top 100: 0
Submission written to ./submission.csv

Top 5 candidates:
  #1 CAND_0018499 | Senior Machine Learning Engineer @ Zomato | 7.2yrs | score=1.0181
  #2 CAND_0052328 | Recommendation Systems Engineer @ Amazon | 6.5yrs | score=1.0166
  ...
```

### 5. Validate before submitting

```bash
python validate_submission.py submission.csv
# → Submission is valid.
```

---

## Project Structure

```
redrob-ranker/
├── rank.py                        # Main ranker — run this
├── validate_submission.py         # Official format validator (from hackathon bundle)
├── requirements.txt               # Dependencies
├── submission_metadata.yaml       # Submission metadata
├── README.md                      # This file
└── sample_output/
    └── submission.csv             # Sample output (format reference)
```

---

## Compute Constraints

This system is designed to meet the hackathon's hard constraints:

| Constraint | Limit | This system |
|---|---|---|
| Runtime | ≤ 5 min | ~2–3 min on standard CPU |
| Memory | ≤ 16 GB | Streams candidates line-by-line (~500 MB peak) |
| GPU | Not allowed | Pure CPU, no torch/tensorflow |
| Network | Not allowed | Zero external calls during ranking |

---

## AI Tools Declaration

Claude was used for architecture discussion and code review. GitHub Copilot was used for autocomplete. No candidate data was fed to any LLM. All scoring logic and design decisions are original engineering work.

---

## Author

**Bandla Hima Naga Sri Harshitha**  
BITS Pilani, Hyderabad Campus  
Dual Degree: B.E. Electronics & Instrumentation + M.Sc. Mathematics
