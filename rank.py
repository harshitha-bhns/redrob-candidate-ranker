#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranker
Author: Bandla Hima Naga Sri Harshitha

Scoring approach:
  1. Honeypot detection  → immediate disqualification (score = 0)
  2. Career history match → most important signal (40%)
  3. Skills match         → weighted by trust (25%)
  4. Experience years     → bell curve around 6-8 yrs (15%)
  5. Location/logistics   → India + notice period (10%)
  6. Behavioral signals   → availability multiplier (10%)

No API calls. No GPU. Runs on CPU in < 5 minutes for 100K candidates.
"""

import json
import csv
import argparse
import math
from datetime import date, datetime
from pathlib import Path

# ─────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────

REFERENCE_DATE = date(2026, 6, 10)

# Skills the JD explicitly requires or values
CORE_SKILLS = {
    # Must-have (high weight)
    "embeddings": 10,
    "vector database": 10,
    "retrieval": 9,
    "ranking": 9,
    "semantic search": 9,
    "sentence transformers": 9,
    "faiss": 9,
    "pinecone": 8,
    "weaviate": 8,
    "qdrant": 8,
    "milvus": 8,
    "elasticsearch": 8,
    "opensearch": 8,
    "hybrid search": 9,
    "information retrieval": 9,
    "nlp": 8,
    "natural language processing": 8,
    "python": 7,
    "machine learning": 7,
    "ml": 7,
    "llm": 8,
    "large language models": 8,
    "fine-tuning": 7,
    "fine tuning": 7,
    "lora": 7,
    "qlora": 7,
    "rag": 8,
    "retrieval augmented generation": 8,
    "transformers": 7,
    "hugging face": 7,
    "huggingface": 7,
    "recommendation systems": 7,
    "search": 6,
    "ranking systems": 8,
    "learning to rank": 8,
    "xgboost": 5,
    "ndcg": 7,
    "mrr": 6,
    "a/b testing": 6,
    "evaluation": 6,
    "bge": 7,
    "e5": 6,
    "openai embeddings": 7,
    # Nice-to-have (lower weight)
    "deep learning": 5,
    "pytorch": 5,
    "tensorflow": 4,
    "distributed systems": 4,
    "inference optimization": 4,
    "open source": 3,
}

# AI/ML job title keywords that indicate real ML work
ML_TITLE_KEYWORDS = [
    "machine learning", "ml engineer", "ai engineer", "applied scientist",
    "data scientist", "nlp engineer", "research engineer", "applied ml",
    "ai researcher", "search engineer", "ranking engineer", "recommendation",
    "applied research", "ml platform", "ai platform", "mlops",
    "senior engineer", "staff engineer", "principal engineer",  # context-dependent
]

# These title keywords are red flags (non-technical)
NON_TECH_TITLE_KEYWORDS = [
    "marketing", "sales", "operations", "hr ", "recruiter", "accountant",
    "finance", "customer support", "customer service", "content writer",
    "seo", "social media", "business development", "project manager",
    "product manager",  # PM is borderline — handled carefully below
]

# IT services companies that the JD explicitly disfavors
IT_SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "mindtree", "l&t infotech", "niit technologies", "mastech",
}

# India-preferred locations
INDIA_PREFERRED_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "delhi", "gurugram", "gurgaon", "chennai", "kolkata",
    "india", "ncr", "delhi ncr",
}

# Proficiency weights
PROFICIENCY_WEIGHT = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.45,
    "beginner": 0.15,
}


# ─────────────────────────────────────────────
# 2. HONEYPOT DETECTION
# ─────────────────────────────────────────────

def is_honeypot(candidate: dict) -> bool:
    """
    Detect impossible profiles that should be disqualified.
    Returns True if the candidate looks like a honeypot.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Check 1: experience years vs career history duration
    stated_exp = profile.get("years_of_experience", 0)
    total_career_months = sum(
        ch.get("duration_months", 0) for ch in career
    )
    total_career_years = total_career_months / 12.0
    # If stated exp is more than 3 years beyond actual career history — suspicious
    if stated_exp > 0 and total_career_years > 0:
        if stated_exp > total_career_years + 3:
            return True

    # Check 2: career at company that couldn't have existed that long
    # (company founded after start_date — we detect via impossible dates)
    for ch in career:
        start_str = ch.get("start_date", "")
        if start_str:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d").date()
                # Start date in the future
                if start > REFERENCE_DATE:
                    return True
                # Start date impossibly far in the past for a typical career
                if start.year < 1990 and stated_exp < 40:
                    return True
            except ValueError:
                pass
        dur = ch.get("duration_months", 0)
        if dur < 0:
            return True

    # Check 3: expert in many skills with 0 duration months
    expert_zero_duration = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero_duration >= 4:
        return True

    # Check 4: impossibly many skills with same duration (copy-paste signal)
    if len(skills) > 5:
        durations = [s.get("duration_months", 0) for s in skills]
        # All non-zero durations are identical — suspicious
        nonzero = [d for d in durations if d > 0]
        if len(nonzero) >= 8 and len(set(nonzero)) == 1:
            return True

    return False


# ─────────────────────────────────────────────
# 3. CAREER HISTORY SCORER
# ─────────────────────────────────────────────

def score_career(candidate: dict) -> float:
    """
    Score based on actual career history — the anti-keyword-stuffer signal.
    Checks: ML/AI titles, product company experience, career trajectory.
    Returns 0.0 – 1.0
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "").lower()
    current_industry = profile.get("current_industry", "").lower()

    # Red flag: current title is non-technical
    for bad in NON_TECH_TITLE_KEYWORDS:
        if bad in current_title:
            return 0.05  # not zero — they might have changed roles

    score = 0.0
    ml_months = 0
    product_company_months = 0
    total_months = 0

    for ch in career:
        title = ch.get("title", "").lower()
        company = ch.get("company", "").lower()
        industry = ch.get("industry", "").lower()
        description = ch.get("description", "").lower()
        dur = ch.get("duration_months", 0)
        is_current = ch.get("is_current", False)
        total_months += dur

        # Check if this role involved ML/AI work
        role_is_ml = False
        for kw in ML_TITLE_KEYWORDS:
            if kw in title:
                role_is_ml = True
                break

        # Also check description for ML work even if title doesn't say so
        ml_desc_signals = [
            "embedding", "retrieval", "ranking", "vector", "semantic",
            "recommendation", "search system", "nlp", "language model",
            "fine-tun", "bert", "transformer", "machine learning model",
            "ml model", "neural network", "deep learning", "a/b test",
        ]
        desc_has_ml = sum(1 for sig in ml_desc_signals if sig in description) >= 2

        if role_is_ml or desc_has_ml:
            ml_months += dur

        # Product company check (not IT services)
        is_it_services = any(s in company for s in IT_SERVICES_COMPANIES)
        is_it_industry = any(x in industry for x in ["it services", "consulting", "outsourcing"])
        if not is_it_services and not is_it_industry:
            product_company_months += dur

        # Recent/current role bonus
        recency_boost = 1.3 if is_current else 1.0

        # Build partial score per role
        if role_is_ml and not is_it_services:
            score += dur * 0.015 * recency_boost  # product ML role — best signal
        elif role_is_ml and is_it_services:
            score += dur * 0.007 * recency_boost  # ML role at services — half credit
        elif desc_has_ml and not is_it_services:
            score += dur * 0.008 * recency_boost  # non-ML title but did ML work

    # Ratio of career spent in ML roles at product companies
    if total_months > 0:
        ml_ratio = ml_months / total_months
        product_ratio = product_company_months / total_months
        score += ml_ratio * 0.3
        score += product_ratio * 0.2

    return min(score, 1.0)


# ─────────────────────────────────────────────
# 4. SKILLS SCORER
# ─────────────────────────────────────────────

def score_skills(candidate: dict) -> float:
    """
    Score based on relevant skills with trust weighting.
    Trust = proficiency × duration_months × endorsements (prevents keyword stuffing).
    Returns 0.0 – 1.0
    """
    skills = candidate.get("skills", [])
    sig = candidate.get("redrob_signals", {})
    assessment_scores = sig.get("skill_assessment_scores", {})

    total_score = 0.0
    max_possible = 0.0

    for skill in skills:
        name = skill.get("name", "").lower().strip()
        proficiency = skill.get("proficiency", "beginner")
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        # Find if this skill matches any core skill
        skill_weight = 0
        for core, weight in CORE_SKILLS.items():
            if core in name or name in core:
                skill_weight = max(skill_weight, weight)
                break

        if skill_weight == 0:
            continue

        # Trust multiplier: proficiency × duration (capped) × endorsement signal
        prof_mult = PROFICIENCY_WEIGHT.get(proficiency, 0.1)
        dur_mult = min(duration / 24.0, 1.0)  # normalize to 2 years max
        end_mult = min(1.0 + endorsements / 50.0, 1.5)  # endorsements boost up to 1.5x

        # Assessment score bonus if available
        assess_boost = 1.0
        for assess_name, assess_score in assessment_scores.items():
            if any(c in assess_name.lower() for c in name.split()):
                assess_boost = 1.0 + (assess_score / 200.0)  # up to 1.5x
                break

        trust = prof_mult * dur_mult * end_mult * assess_boost

        # Penalize expert + 0 duration (keyword stuffing signal)
        if proficiency == "expert" and duration == 0:
            trust *= 0.1

        total_score += skill_weight * trust
        max_possible += skill_weight * 1.5 * 1.5  # theoretical max trust

    if max_possible == 0:
        return 0.0
    return min(total_score / max_possible, 1.0)


# ─────────────────────────────────────────────
# 5. EXPERIENCE YEARS SCORER
# ─────────────────────────────────────────────

def score_experience(candidate: dict) -> float:
    """
    Bell curve around 6-8 years. JD says 5-9 is target.
    Returns 0.0 – 1.0
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    if yoe <= 0:
        return 0.0
    elif yoe < 3:
        return 0.1  # too junior
    elif yoe < 5:
        return 0.3 + (yoe - 3) * 0.15  # 0.3 to 0.6
    elif yoe <= 9:
        # Peak zone: 5-9 years, sweet spot at 6-8
        peak = 7.0
        spread = 2.0
        return 0.7 + 0.3 * math.exp(-((yoe - peak) ** 2) / (2 * spread ** 2))
    elif yoe <= 12:
        return 0.55 - (yoe - 9) * 0.05  # 0.55 to 0.40
    else:
        return max(0.25, 0.4 - (yoe - 12) * 0.03)  # diminishing returns for very senior


# ─────────────────────────────────────────────
# 6. LOCATION + LOGISTICS SCORER
# ─────────────────────────────────────────────

def score_location_logistics(candidate: dict) -> float:
    """
    Score based on location preference, notice period, and work mode.
    Returns 0.0 – 1.0
    """
    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})

    score = 0.0
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_relocate = sig.get("willing_to_relocate", False)
    notice_days = sig.get("notice_period_days", 90)
    work_mode = sig.get("preferred_work_mode", "")

    # Location scoring
    if country == "india":
        score += 0.4
        # Preferred cities get extra
        for city in INDIA_PREFERRED_CITIES:
            if city in location:
                score += 0.2
                break
    elif willing_relocate:
        score += 0.15  # outside India but willing to relocate
    else:
        score += 0.05  # outside India, not relocating

    # Notice period
    if notice_days <= 30:
        score += 0.25
    elif notice_days <= 60:
        score += 0.15
    elif notice_days <= 90:
        score += 0.08
    else:
        score += 0.0  # 90+ days is a real concern per JD

    # Work mode (hybrid preferred per JD)
    if work_mode in ("hybrid", "flexible"):
        score += 0.1
    elif work_mode == "onsite":
        score += 0.08
    else:
        score += 0.03  # remote only — harder logistically

    return min(score, 1.0)


# ─────────────────────────────────────────────
# 7. BEHAVIORAL SIGNALS MULTIPLIER
# ─────────────────────────────────────────────

def behavioral_multiplier(candidate: dict) -> float:
    """
    A multiplier (0.3 – 1.2) based on platform engagement signals.
    A perfect-on-paper candidate who's inactive gets down-weighted.
    Returns 0.3 – 1.2
    """
    sig = candidate.get("redrob_signals", {})

    multiplier = 1.0

    # Recency of activity
    last_active_str = sig.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_inactive = (REFERENCE_DATE - last_active).days
            if days_inactive <= 30:
                multiplier += 0.10
            elif days_inactive <= 90:
                multiplier += 0.05
            elif days_inactive <= 180:
                multiplier -= 0.10
            else:
                multiplier -= 0.25  # inactive > 6 months
        except ValueError:
            pass

    # Open to work flag
    if sig.get("open_to_work_flag", False):
        multiplier += 0.08

    # Recruiter response rate
    response_rate = sig.get("recruiter_response_rate", 0.5)
    if response_rate >= 0.7:
        multiplier += 0.08
    elif response_rate >= 0.4:
        multiplier += 0.03
    elif response_rate < 0.15:
        multiplier -= 0.15  # ghost candidate

    # Interview completion rate
    interview_rate = sig.get("interview_completion_rate", 0.5)
    if interview_rate >= 0.8:
        multiplier += 0.05
    elif interview_rate < 0.3:
        multiplier -= 0.10

    # GitHub activity (relevant for engineering roles)
    github = sig.get("github_activity_score", -1)
    if github >= 60:
        multiplier += 0.07
    elif github >= 30:
        multiplier += 0.03
    elif github == -1:
        pass  # no GitHub — neutral, not negative

    # Profile completeness
    completeness = sig.get("profile_completeness_score", 50)
    if completeness >= 85:
        multiplier += 0.05
    elif completeness < 40:
        multiplier -= 0.05

    return max(0.3, min(multiplier, 1.3))


# ─────────────────────────────────────────────
# 8. REASONING GENERATOR
# ─────────────────────────────────────────────

def generate_reasoning(candidate: dict, component_scores: dict, rank: int) -> str:
    """
    Generate specific, honest, rank-consistent reasoning.
    References actual facts from the candidate's profile.
    """
    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "")
    country = profile.get("country", "")

    # Find top relevant skills this candidate has
    relevant_skills = []
    for s in skills:
        name_lower = s.get("name", "").lower()
        for core in CORE_SKILLS:
            if core in name_lower or name_lower in core:
                relevant_skills.append(s.get("name"))
                break
    relevant_skills = relevant_skills[:3]

    # Most recent relevant role
    recent_ml_role = None
    for ch in sorted(career, key=lambda x: x.get("start_date", ""), reverse=True):
        title_lower = ch.get("title", "").lower()
        desc_lower = ch.get("description", "").lower()
        if any(kw in title_lower for kw in ["ml", "ai", "machine learning", "data scientist", "nlp"]):
            recent_ml_role = ch
            break
        if any(sig in desc_lower for sig in ["embedding", "retrieval", "ranking", "vector", "recommendation"]):
            recent_ml_role = ch
            break

    notice = sig.get("notice_period_days", 90)
    response_rate = sig.get("recruiter_response_rate", 0)
    last_active = sig.get("last_active_date", "")
    open_to_work = sig.get("open_to_work_flag", False)

    # Build reasoning parts
    parts = []

    # Opening: who they are
    if recent_ml_role:
        parts.append(
            f"{title} with {yoe:.0f} yrs exp; "
            f"recent ML work as {recent_ml_role['title']} at {recent_ml_role['company']}"
        )
    else:
        parts.append(f"{title} at {company} with {yoe:.0f} yrs experience")

    # Skills highlight
    if relevant_skills:
        parts.append(f"relevant skills: {', '.join(relevant_skills)}")

    # Location / logistics
    loc_str = f"{location}, {country}" if location else country
    if loc_str:
        parts.append(f"based in {loc_str}")
    if notice <= 30:
        parts.append(f"notice period {notice}d")
    elif notice > 90:
        parts.append(f"notice {notice}d is a concern")

    # Engagement
    if not open_to_work and rank <= 30:
        parts.append("not marked open-to-work")
    if response_rate < 0.2 and rank <= 50:
        parts.append(f"low recruiter response rate ({response_rate:.0%})")
    if last_active:
        try:
            la = datetime.strptime(last_active, "%Y-%m-%d").date()
            days = (REFERENCE_DATE - la).days
            if days > 180:
                parts.append(f"inactive {days}d — availability concern")
        except ValueError:
            pass

    # Honest gap for lower ranks
    if rank > 50 and component_scores.get("career", 0) < 0.2:
        parts.append("limited ML career history")
    if rank > 70 and component_scores.get("skills", 0) < 0.15:
        parts.append("weak match on core skills")

    return "; ".join(parts)[:300]  # keep it tight


# ─────────────────────────────────────────────
# 9. COMPOSITE SCORER
# ─────────────────────────────────────────────

WEIGHTS = {
    "career":    0.40,
    "skills":    0.25,
    "experience": 0.15,
    "location":  0.10,
    "behavioral": 0.10,
}


def score_candidate(candidate: dict) -> tuple[float, dict]:
    """
    Compute composite score for a candidate.
    Returns (final_score, component_scores_dict)
    """
    if is_honeypot(candidate):
        return 0.0, {"honeypot": True}

    career_s = score_career(candidate)
    skills_s = score_skills(candidate)
    exp_s = score_experience(candidate)
    loc_s = score_location_logistics(candidate)

    # Weighted sum (behavioral is a multiplier, not additive)
    base = (
        WEIGHTS["career"]     * career_s +
        WEIGHTS["skills"]     * skills_s +
        WEIGHTS["experience"] * exp_s +
        WEIGHTS["location"]   * loc_s
    )

    # Behavioral signals applied as a multiplier on 90% of score
    b_mult = behavioral_multiplier(candidate)
    final = base * 0.90 * b_mult + base * 0.10

    components = {
        "career": career_s,
        "skills": skills_s,
        "experience": exp_s,
        "location": loc_s,
        "behavioral_mult": b_mult,
    }
    return round(final, 6), components


# ─────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────

def run(candidates_path: str, output_path: str, participant_id: str = "submission"):
    print(f"Loading candidates from {candidates_path}...")

    scored = []
    honeypot_count = 0
    total = 0

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            total += 1

            score, components = score_candidate(candidate)

            if components.get("honeypot"):
                honeypot_count += 1

            scored.append({
                "candidate_id": candidate["candidate_id"],
                "score": score,
                "components": components,
                "candidate": candidate,
            })

            if total % 10000 == 0:
                print(f"  Processed {total:,} candidates...")

    print(f"\nTotal: {total:,} | Honeypots detected: {honeypot_count}")

    # Sort by score descending, break ties by candidate_id ascending
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Take top 100
    top100 = scored[:100]

    # Check honeypot rate in top 100
    hp_in_top = sum(1 for c in top100 if c["components"].get("honeypot"))
    print(f"Honeypots in top 100: {hp_in_top} ({hp_in_top}% — must be ≤10%)")

    # Write CSV
    out_file = Path(output_path)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_idx, item in enumerate(top100, start=1):
            reasoning = generate_reasoning(
                item["candidate"],
                item["components"],
                rank_idx
            )
            writer.writerow([
                item["candidate_id"],
                rank_idx,
                item["score"],
                reasoning,
            ])

    print(f"\nSubmission written to {out_file}")
    print(f"Top 5 candidates:")
    for i, item in enumerate(top100[:5], 1):
        p = item["candidate"]["profile"]
        print(f"  #{i} {item['candidate_id']} | {p['current_title']} @ {p['current_company']} "
              f"| {p['years_of_experience']}yrs | score={item['score']:.4f}")


# ─────────────────────────────────────────────
# 11. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob candidate ranker")
    parser.add_argument(
        "--candidates",
        default="./candidates.jsonl",
        help="Path to candidates.jsonl (default: ./candidates.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="./submission.csv",
        help="Output CSV path (default: ./submission.csv)",
    )
    parser.add_argument(
        "--id",
        default="submission",
        help="Your participant/team ID (used in output filename if --out not set)",
    )
    args = parser.parse_args()

    run(args.candidates, args.out, args.id)
