# ingest_degree_plans.py
# Antikythera Pipeline — LSCO ABC NCCER Alignment
# Degree plan HTML ingest: extract course lists from LSCO catalog HTML files.
# Produces structured records: program, semester, rubric, number, title, credits.
# Choose One clusters preserved with all options listed.

import re
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
from code.config import DEGREE_PLANS, PROCESSED_DIR, BCM_ALLOWED_RUBRICS

# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────

# Course line: "RUBRIC NNNN - Title Credits: N"
RE_COURSE = re.compile(
    r"([A-Z]{2,6})\s+(\d{4})[\s\xa0]*-[\s\xa0]*(.+?)\s+Credits:\s*(\d+)"
)

# Semester header
RE_SEMESTER = re.compile(
    r"^(First|Second|Third|Fourth|Fifth|Sixth)\s+Semester", re.IGNORECASE
)

# Semester hours line
RE_SEM_HOURS = re.compile(
    r"Semester Hours:\s*(\d+)", re.IGNORECASE
)

# Total program hours
RE_TOTAL_HOURS = re.compile(
    r"Total Program Hours:\s*(\d+)", re.IGNORECASE
)

# Choose One marker
RE_CHOOSE_ONE = re.compile(
    r"Choose One", re.IGNORECASE
)


# ─────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────

def parse_degree_plan_html(filepath: Path, program_key: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    text = soup.get_text()

    # Find content start
    start_markers = ["First Semester", "Semester 1", "First Year"]
    start_idx = -1
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = idx
            break

    if start_idx == -1:
        print(f"  WARNING: Could not find semester content in {filepath.name}")
        return []

    # Find content end
    end_markers = ["Total Program Hours", "Return to:", "Next Steps"]
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker, start_idx)
        if idx != -1:
            end_idx = min(end_idx, idx + 50)
            break

    content = text[start_idx:end_idx]

    # Determine semester for each course by finding last semester
    # header before each course match position
    semester_markers = [
        "First Semester", "Second Semester", "Third Semester",
        "Fourth Semester", "Fifth Semester", "Sixth Semester"
    ]

    # Build list of (position, semester_name) for all semester headers
    semester_positions = []
    for sem in semester_markers:
        idx = 0
        while True:
            pos = content.find(sem, idx)
            if pos == -1:
                break
            semester_positions.append((pos, sem))
            idx = pos + 1
    semester_positions.sort(key=lambda x: x[0])

    def get_semester_at(pos):
        current = "Semester 1"
        for sem_pos, sem_name in semester_positions:
            if sem_pos <= pos:
                current = sem_name
            else:
                break
        return current

    # Determine choose_one status at each position
    choose_one_positions = []
    idx = 0
    while True:
        pos = content.find("Choose One", idx)
        if pos == -1:
            break
        choose_one_positions.append(pos)
        idx = pos + 1

    sem_hour_positions = []
    for m in RE_SEM_HOURS.finditer(content):
        sem_hour_positions.append(m.start())

    def is_choose_one_at(pos):
        # Find last Choose One before this position
        last_choose = -1
        for cp in choose_one_positions:
            if cp < pos:
                last_choose = cp
        if last_choose == -1:
            return False
        # Check if a Semester Hours line appears between last Choose One and pos
        for sp in sem_hour_positions:
            if last_choose < sp < pos:
                return False
        return True

    courses = []
    sem_num_map = {name: i+1 for i, (_, name) in enumerate(semester_positions)}

    for match in RE_COURSE.finditer(content):
        pos     = match.start()
        rubric  = match.group(1)
        # Strip OR prefix artifact from Choose One separator
        if rubric.startswith("OR"):
            rubric = rubric[2:]
        number  = match.group(2)
        title   = match.group(3).strip()
        credits = int(match.group(4))
        semester = get_semester_at(pos)

        courses.append({
            "program":      program_key,
            "semester":     semester,
            "semester_num": semester_positions.index(
                next((x for x in semester_positions if x[1] == semester), (0, semester)), 
            ) + 1 if semester_positions else 1,
            "rubric":       rubric,
            "number":       number,
            "course_id":    f"{rubric} {number}",
            "title":        title,
            "credits":      credits,
            "choose_one":   is_choose_one_at(pos),
        })

    return courses


# ─────────────────────────────────────────────
# BCM RUBRIC FILTER
# ─────────────────────────────────────────────

def apply_bcm_filter(courses: list[dict]) -> list[dict]:
    """
    BCM AAS included at limited scope — CNBT rubric courses only.
    Filter out all other rubrics for BCM.
    """
    filtered = [
        c for c in courses
        if c["program"] != "bcm_aas"
        or c["rubric"] in BCM_ALLOWED_RUBRICS
    ]
    removed = len(courses) - len(filtered)
    if removed:
        print(f"  BCM filter: removed {removed} non-CNBT courses")
    return filtered


# ─────────────────────────────────────────────
# SAVE TO PROCESSED
# ─────────────────────────────────────────────

def save_processed(courses: list[dict], filename: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outpath = PROCESSED_DIR / filename
    df = pd.DataFrame(courses)
    df.to_csv(outpath, index=False)
    print(f"Saved: {outpath} ({len(df)} records)")
    return outpath


# ─────────────────────────────────────────────
# MAIN INGEST
# ─────────────────────────────────────────────

def run_degree_plan_ingest() -> list[dict]:
    """
    Parse all six degree plan HTML files.
    Saves combined course list to processed/.
    Returns list of all course records.
    """
    print("\n=== DEGREE PLAN INGEST ===")
    all_courses = []

    for program_key, filepath in DEGREE_PLANS.items():
        print(f"\n{program_key}: {filepath.name}")
        courses = parse_degree_plan_html(filepath, program_key)
        print(f"  Courses found: {len(courses)}")
        all_courses.extend(courses)

    all_courses = apply_bcm_filter(all_courses)
    save_processed(all_courses, "degree_plans_processed.csv")
    print(f"\nTotal courses across all programs: {len(all_courses)}")
    print("=== DEGREE PLAN INGEST COMPLETE ===\n")
    return all_courses


if __name__ == "__main__":
    run_degree_plan_ingest()