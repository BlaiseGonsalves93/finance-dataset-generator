#!/usr/bin/env python3
"""
FinBot Dataset Validator
pip install tqdm
python validate_dataset.py --input ../dataset/raw/samples.jsonl --output ../dataset/reviewed/clean.jsonl
"""

import json
import re
import argparse
import hashlib
from pathlib import Path
from collections import Counter
from typing import List, Tuple

BANNED = [
    "guaranteed returns","will definitely give","100% safe",
    "no risk involved","assured profit","sure shot",
    "100% guarantee","promise you returns","certain to profit",
    "cannot lose money","riskless",
]
DISCLAIMER_MARKERS = [
    "consult","sebi","registered","financial advisor",
    "not personalized","educational","risk","disclaimer",
]
MIN_WORDS = 80
MAX_WORDS = 900


def load(path: str) -> List[dict]:
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:    out.append(json.loads(line))
            except: print(f"  Bad JSON line {i+1}")
    return out


def save(samples: List[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} → {path}")


def chk_schema(s):
    for k in ["instruction","input","output"]:
        if k not in s: return False, f"Missing: {k}"
    if not s["output"].strip(): return False, "Empty output"
    return True, "ok"

def chk_length(s):
    w = len(s["output"].split())
    if w < MIN_WORDS: return False, f"Short ({w}w)"
    if w > MAX_WORDS: return False, f"Long ({w}w)"
    return True, "ok"

def chk_banned(s):
    low = s["output"].lower()
    for p in BANNED:
        if p in low: return False, f"Banned: '{p}'"
    return True, "ok"

def chk_disclaimer(s):
    task = s.get("_task","")
    if task in ("refusal","qa"): return True, "ok"
    low = s["output"].lower()
    if not any(m in low for m in DISCLAIMER_MARKERS):
        return False, "No disclaimer"
    return True, "ok"

def chk_india(s):
    low = s["output"].lower()
    markers = ["₹","sebi","amfi","nse","bse","lakh","crore",
               "nifty","sensex","rbi","elss","sip","ltcg"]
    if not any(m in low for m in markers):
        return False, "No Indian context"
    return True, "ok"

def chk_prediction(s):
    low = s["output"].lower()
    for pat in [r"will reach ₹\d+", r"will be at ₹\d+"]:
        if re.search(pat, low): return False, "Price prediction"
    return True, "ok"


CHECKS = [chk_schema, chk_length, chk_banned, chk_disclaimer, chk_india, chk_prediction]


def validate(s: dict) -> Tuple[bool, List[str]]:
    fails = []
    for chk in CHECKS:
        ok, reason = chk(s)
        if not ok: fails.append(reason)
    return len(fails) == 0, fails


def dedup(samples: List[dict]) -> Tuple[List[dict], int]:
    seen, unique, dupes = set(), [], 0
    for s in samples:
        h = hashlib.md5((s.get("input","") + s.get("output","")[:300]).lower().encode()).hexdigest()
        if h in seen: dupes += 1
        else: seen.add(h); unique.append(s)
    return unique, dupes


def report(total, passed, rejected, fails):
    print("\n" + "="*55)
    print("📊 VALIDATION REPORT")
    print("="*55)
    print(f"  Input    : {total}")
    print(f"  ✅ Passed : {len(passed)} ({len(passed)/total*100:.1f}%)")
    print(f"  ❌ Rejected: {len(rejected)} ({len(rejected)/total*100:.1f}%)")
    print("\n  Top rejection reasons:")
    for r, c in Counter(fails).most_common(8):
        print(f"    {r}: {c}")
    if passed:
        print("\n  Task distribution:")
        for t, c in sorted(Counter(s.get("_task","?") for s in passed).items(), key=lambda x:-x[1]):
            print(f"    {t:25} {c:5} ({c/len(passed)*100:.1f}%)")
        lens = [len(s["output"].split()) for s in passed]
        print(f"\n  Output words — min:{min(lens)} max:{max(lens)} mean:{sum(lens)//len(lens)}")
    print("="*55)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",    required=True)
    ap.add_argument("--output",   default="../dataset/reviewed/clean.jsonl")
    ap.add_argument("--rejected", default="../dataset/reviewed/rejected.jsonl")
    args = ap.parse_args()

    print(f"\n🔍 Loading {args.input}")
    samples = load(args.input)
    print(f"   {len(samples)} samples loaded")

    samples, dupes = dedup(samples)
    print(f"   {dupes} duplicates removed → {len(samples)} unique")

    passed, rejected, all_fails = [], [], []
    for s in samples:
        ok, fails = validate(s)
        if ok: passed.append(s)
        else:
            s["_failures"] = fails
            rejected.append(s)
            all_fails.extend(fails)

    save(passed, args.output)
    save(rejected, args.rejected)
    report(len(samples), passed, rejected, all_fails)


if __name__ == "__main__":
    main()