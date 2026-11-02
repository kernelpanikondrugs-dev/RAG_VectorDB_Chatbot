import re
import json

SOURCE_MD = "FAQ_Restructured_QA_Format.md"
OUTPUT_JSON = "kb_pairs.json"

def normalize_text(text: str) -> str:
    """
    Normalizes text by softening harsh ALL-CAPS words into professional, 
    natural case formatting while preserving readability.
    """
    # Replace specific legal/emphasis all-caps terms naturally
    replacements = {
        r'\bNOT\b': 'not',
        r'\bNEVER\b': 'never',
        r'\bALWAYS\b': 'always',
        r'\bDO NOT\b': 'do not',
        r'\bDON\'T\b': "don't",
        r'\bMANDATE\b': 'mandate',
        r'\bATTORNEYS\b': 'attorneys',
        r'\bFEES\b': 'fees',
        r'\bCOSTS\b': 'costs',
        r'\bAFFORD\b': 'afford',
        r'\bBOOSTER LOANS?\b': 'booster loan(s)',
        r'\bCREDITORS?\b': 'creditor(s)',
    }
    
    normalized = text
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
    return normalized.strip()

def categorize_section(section_name: str) -> str:
    sec = section_name.lower()
    if "debt review" in sec:
        return "legal_compliance"
    elif "fees" in sec or "payment" in sec:
        return "fees_and_costs"
    elif "loan" in sec or "booster" in sec:
        return "loan_qualification"
    elif "bank" in sec:
        return "banking_and_debit_orders"
    elif "creditor" in sec or "phone" in sec:
        return "creditor_relations"
    elif "score" in sec or "interest" in sec:
        return "credit_score_rehabilitation"
    else:
        return "general_program_info"

def build_keywords(text: str) -> list:
    stop_words = {"what", "when", "where", "which", "who", "whom", "this", "that", "there", "have", "with", "your", "from", "they", "will"}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return list(set([w for w in words if w not in stop_words]))[:8]

def convert_md_to_vector_json():
    with open(SOURCE_MD, "r", encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r'={5,}\nSECTION \d+:\s*(.*?)\n={5,}', content)
    
    kb_entries = []
    chunk_counter = 1

    for i in range(1, len(sections), 2):
        section_title = sections[i].strip()
        section_body = sections[i+1]
        current_category = categorize_section(section_title)

        qa_pairs = re.findall(r'Q:\s*(.*?)\nA:\s*(.*?)(?=\n\nQ:|\n={5,}|$)', section_body, re.DOTALL)

        for q, a in qa_pairs:
            clean_q = normalize_text(q.strip())
            clean_a = normalize_text(a.strip())

            chunk = {
                "id": f"kb_chunk_{chunk_counter:03d}",
                "category": current_category,
                "question": clean_q,
                "answer": clean_a,
                "content": f"Question: {clean_q}\nAnswer: {clean_a}",
                "keywords": build_keywords(f"{clean_q} {clean_a}"),
                "variations": [clean_q]
            }
            kb_entries.append(chunk)
            chunk_counter += 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(kb_entries, f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully normalized and converted {len(kb_entries)} Q&A pairs into '{OUTPUT_JSON}'!")

if __name__ == "__main__":
    convert_md_to_vector_json()