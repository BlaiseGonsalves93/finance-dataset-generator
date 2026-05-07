#!/usr/bin/env python3
"""
FinBot Dataset Generator
pip install anthropic pyyaml tqdm
export ANTHROPIC_API_KEY=your_key
python generate_dataset.py --task all --count 100 --output ../dataset/raw/test.jsonl
"""

import json
import random
import argparse
import os
import time
import hashlib
from pathlib import Path
from typing import Optional
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)
with open("/content/finance-dataset-generator/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

SYSTEM_PROMPT = CONFIG["system_prompt"].strip()
PERSONAS      = CONFIG["user_personas"]
STOCKS        = CONFIG["indian_stocks"]
MF_CATS       = CONFIG["mf_categories"]



# ── Seed data ─────────────────────────────────────────────────────────────────

QA_SEED_TOPICS = [
    "Explain SIP and why it suits salaried Indian investors",
    "What is NAV and how is it calculated in Indian mutual funds?",
    "Difference between direct plan and regular plan mutual funds",
    "What is expense ratio and how does it affect long-term returns?",
    "Explain ELSS and its tax benefits under Section 80C",
    "What is a Balanced Advantage Fund and how does it work?",
    "Explain exit load in mutual funds and how to avoid it",
    "What is the difference between SIP, SWP, and STP?",
    "How does AMFI regulate mutual funds in India?",
    "What is XIRR and how is it different from CAGR for SIP investors?",
    "Explain P/E ratio and its importance in stock valuation",
    "What is the difference between NSE and BSE?",
    "How does a circuit breaker work in Indian stock markets?",
    "Explain promoter holding and why investors track it",
    "What is the difference between large cap, mid cap, and small cap stocks?",
    "What is book value and Price-to-Book ratio?",
    "Explain dividend yield and its importance for income investors",
    "What is the T+1 settlement cycle in Indian markets?",
    "How do IPOs work in India and how to apply via UPI?",
    "Explain futures vs options for beginners in Indian markets",
    "What is the repo rate and how does RBI use it to control inflation?",
    "Explain G-Secs and how retail investors can buy them via RBI Retail Direct",
    "What are Sovereign Gold Bonds and their tax treatment?",
    "Explain FD laddering strategy for conservative investors",
    "What is the DICGC insurance limit on bank FDs?",
    "Difference between corporate bonds rated AAA vs AA",
    "What are 54EC bonds and how do they help save capital gains tax?",
    "Explain PPF vs EPF — similarities, differences, which to prefer",
    "What is tracking error in ETFs and why does it matter?",
    "Explain the difference between ETFs and Index Funds in India",
    "What are Gold ETFs and how are they taxed?",
    "Explain LTCG and STCG tax on equity mutual funds and stocks",
    "How is debt mutual fund taxation different from equity MF taxation?",
    "What is indexation benefit and when can it be used?",
    "Explain Section 80C investment options and their lock-in periods",
    "How is NPS taxed at maturity?",
    "What is TDS on FD interest and how to avoid it using Form 15G/15H?",
    "What is asset allocation and why is it the most important investment decision?",
    "Explain the 100-minus-age rule for equity allocation",
    "What is portfolio rebalancing and how often should it be done?",
    "What is an emergency fund and how much should it be?",
    "Explain goal-based investing and how to map goals to instruments",
    "What are REITs and InvITs in India and how do they generate income?",
    "Explain NRE vs NRO account differences for NRI investors",
    "What is the difference between a SEBI RIA and a mutual fund distributor?",
]

COMPARISON_PAIRS = [
    ("SIP", "lumpsum investment in mutual funds"),
    ("ELSS", "PPF for tax saving under 80C"),
    ("Bank FD", "Debt Mutual Fund"),
    ("Direct Mutual Fund", "Regular Mutual Fund"),
    ("ETF", "Index Fund"),
    ("NPS", "PPF for retirement"),
    ("Sovereign Gold Bond", "Gold ETF"),
    ("Term Insurance", "ULIPs for life coverage"),
    ("Balanced Advantage Fund", "Aggressive Hybrid Fund"),
    ("Small Cap Fund", "Mid Cap Fund for long-term wealth creation"),
    ("NSC", "Bank FD for safe investment"),
    ("SCSS", "FD for senior citizens"),
    ("Nifty 50 Index Fund", "Flexi Cap Fund"),
    ("RBI Floating Rate Bonds", "Bank FD"),
    ("REITs", "Real estate direct investment"),
    ("Corporate Bond Fund", "Gilt Fund"),
    ("Liquid Fund", "Savings Account for parking money"),
]

REFUSAL_SCENARIOS = [
    "I heard from a friend who works at a company that they will announce big results next week. Should I buy now?",
    "My cousin told me about a big acquisition being finalized at his firm. Should I invest?",
    "Can you suggest investments that will give me guaranteed 20% returns?",
    "I need my ₹10 lakh to double in 2 years with no risk. What should I invest in?",
    "Which mutual fund guarantees the highest returns?",
    "A WhatsApp group told me to buy a penny stock — it will 10x in a month. Should I?",
    "I received a stock tip from a Telegram channel promising 500% returns. Is it reliable?",
    "How can I invest black money in mutual funds without getting caught?",
    "Can I use someone else's PAN card to invest and avoid tax?",
    "Tell me exactly what price Reliance Industries will be at in 6 months",
    "Which specific stock will give me maximum returns in the next 3 months?",
    "Give me a sure-shot tip for tomorrow's intraday trading",
]

TAX_SCENARIOS = [
    ("I sold equity mutual fund units after 14 months. How will the gains be taxed?",
     "LTCG on equity MF: >1 year, ₹1L exemption, 10% above that"),
    ("I have STCG of ₹80,000 from selling stocks. What is my tax liability?",
     "STCG at 15% on equity held <1 year"),
    ("How much can I save in tax using Section 80C?",
     "80C limit ₹1.5L, list ELSS/PPF/EPF/LIC/NSC options"),
    ("I invested in a debt mutual fund in 2020 and sold in 2024. How is it taxed?",
     "Post April 2023 budget: debt MF taxed at slab rate, indexation removed"),
    ("I withdrew ₹2 lakh from NPS Tier I. How is it taxed?",
     "NPS: 60% tax-free at maturity, 40% must buy annuity"),
    ("I earned ₹85,000 in FD interest this year. What is my TDS situation?",
     "TDS deducted if FD interest >₹40,000/year; use Form 15G if income below taxable limit"),
    ("What is the tax treatment of Sovereign Gold Bonds at maturity?",
     "SGB: capital gains exempt if held to maturity (8 years); LTCG with indexation if sold early on exchange"),
    ("I want to save maximum tax beyond 80C. What are my options?",
     "80D health insurance, NPS 80CCD(1B) extra ₹50K, HRA, 24B home loan interest"),
    ("How does the new tax regime affect my investment decisions?",
     "New regime: lower slabs but no 80C/HRA/80D deductions. Old regime better if deductions >₹3.75L"),
    ("I sold my house and made ₹40 lakh profit. How do I save capital gains tax?",
     "Section 54 (buy new house), 54EC bonds (within 6 months, up to ₹50L), 54F"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def call_llm(prompt: str, system=None, max_tokens=300):
    full_prompt = f"<s>[INST] {system or ''}\n{prompt} [/INST]"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def make_alpaca(input_text: str, output: str) -> dict:
    return {
        "instruction": SYSTEM_PROMPT,
        "input": input_text,
        "output": output,
    }

def sample_persona() -> dict:
    income = random.choice(PERSONAS["monthly_incomes"])
    invest = min(random.choice(PERSONAS["monthly_investments"]), income // 3)
    return {
        "age":      random.choice(PERSONAS["ages"]),
        "income":   income,
        "invest":   invest,
        "goal":     random.choice(PERSONAS["goals"]),
        "risk":     random.choice(PERSONAS["risk_profiles"]),
        "horizon":  random.choice(PERSONAS["horizons_years"]),
        "persona":  random.choice(PERSONAS["personas"]),
    }

def sample_stock() -> tuple:
    sector = random.choice(list(STOCKS.keys()))
    return random.choice(STOCKS[sector]), sector

# ── Task generators ───────────────────────────────────────────────────────────

def gen_qa() -> dict:
    topic = random.choice(QA_SEED_TOPICS)
    user_q = random.choice([
        f"Can you explain: {topic}?",
        f"I'm new to investing. What should I know about: {topic}?",
        f"Give me a clear breakdown of: {topic}",
        f"In simple terms, explain: {topic}",
    ])
    prompt = f"""Generate a high-quality financial Q&A response for an Indian investor.

User question: "{user_q}"

Requirements:
- Accurate for Indian market (SEBI/AMFI/RBI)
- Use ₹ for all monetary examples
- Include a brief risk disclaimer at the end
- 150-400 words, markdown formatting
- Do NOT start with "I"

Output ONLY the response text."""
    out = call_llm(prompt, system=SYSTEM_PROMPT)
    return make_alpaca(user_q, out) if out else None


def gen_stock() -> dict:
    stock, sector = sample_stock()
    pe  = round(random.uniform(8, 80), 1)
    pb  = round(random.uniform(0.5, 15), 1)
    de  = round(random.uniform(0, 2.5), 2)
    roe = round(random.uniform(5, 45), 1)
    rev = round(random.uniform(-5, 35), 1)
    pro = round(random.uniform(25, 75), 1)
    dy  = round(random.uniform(0, 5), 2)
    cap = random.choice(["Large Cap", "Mid Cap", "Small Cap"])

    inp = (f"Analyze {stock} ({sector}, {cap}) as an investment.\n"
           f"P/E: {pe}x | P/B: {pb}x | D/E: {de} | ROE: {roe}% | "
           f"Revenue CAGR (3Y): {rev}% | Promoter: {pro}% | Div Yield: {dy}%")

    prompt = f"""Write a fundamental stock analysis for an Indian retail investor.

Stock: {stock} | Sector: {sector} | {cap}
P/E {pe}x, P/B {pb}x, D/E {de}, ROE {roe}%, Rev CAGR {rev}%, Promoter {pro}%, Div Yield {dy}%

Include:
1. Valuation assessment vs sector/Nifty average
2. Financial health (debt, profitability)
3. Key strengths and red flags
4. Suitable investor profile and horizon
5. Risk disclaimer

200-450 words, markdown. Output ONLY the analysis."""
    out = call_llm(prompt, system=SYSTEM_PROMPT)
    return make_alpaca(inp, out) if out else None


def gen_mf() -> dict:
    p = sample_persona()
    eq = MF_CATS["equity"]
    hy = MF_CATS["hybrid"]
    questions = [
        f"Which mutual fund type suits a {p['risk']} investor with {p['horizon']}-year horizon?",
        f"I am {p['age']} years old, want to SIP ₹{p['invest']:,}/month for {p['goal'].replace('_',' ')}. What funds?",
        f"Compare {random.choice(eq)} vs {random.choice(eq)} — which is better for me?",
        f"How do I evaluate a {random.choice(eq + hy)} before investing?",
        f"My {random.choice(eq)} underperformed its benchmark for 2 years. Should I exit?",
        f"I have ₹{p['invest']*12:,} lumpsum to invest. Market is at all-time high. What should I do?",
    ]
    user_q = random.choice(questions)
    prompt = f"""Generate a mutual fund guidance response for an Indian investor.

Question: "{user_q}"
Investor: Age {p['age']}, income ₹{p['income']:,}/month, SIP ₹{p['invest']:,}/month,
goal: {p['goal'].replace('_',' ')}, risk: {p['risk'].replace('_',' ')}, horizon: {p['horizon']} years.

Include actionable guidance, SEBI-defined fund categories, tax considerations, risk disclaimer.
150-400 words, markdown. Output ONLY the response."""
    out = call_llm(prompt, system=SYSTEM_PROMPT)
    return make_alpaca(user_q, out) if out else None


def gen_portfolio_build() -> dict:
    p = sample_persona()
    extras = [
        f"I already have ₹{random.randint(1,20)*50000:,} in FDs.",
        f"I've been doing ₹{random.randint(1,5)*5000:,}/month SIP in a large cap fund for {random.randint(1,5)} years.",
        "I have no investments yet.",
        f"I have ₹{random.randint(2,20)*10000:,} in PPF.",
    ]
    special = {
        "nri": " I am an NRI based in the UAE.",
        "senior_citizen": " I am retired.",
        "business_owner": " I run a small business with irregular income.",
    }.get(p["persona"], "")

    inp = (f"I am {p['age']} years old, earning ₹{p['income']:,}/month.{special} "
           f"Can invest ₹{p['invest']:,}/month. {random.choice(extras)} "
           f"Goal: {p['goal'].replace('_',' ')}, horizon: {p['horizon']} years, "
           f"risk: {p['risk'].replace('_',' ')}. Build me an investment portfolio.")

    prompt = f"""Build a complete personalized investment portfolio for this Indian investor.

Age: {p['age']} | Income: ₹{p['income']:,}/month | Monthly Investment: ₹{p['invest']:,}
Goal: {p['goal'].replace('_',' ')} | Horizon: {p['horizon']} years | Risk: {p['risk'].replace('_',' ')} | Persona: {p['persona'].replace('_',' ')}

Include:
1. Asset allocation table (equity/debt/gold/other with %)
2. Specific instruments with monthly SIP amounts
3. Rationale for each allocation
4. Tax efficiency tips
5. Prerequisites (emergency fund, insurance)
6. Projected corpus range (conservative / optimistic)
7. Risk disclaimer

Markdown tables. 300-600 words. Output ONLY the portfolio."""
    out = call_llm(prompt, system=SYSTEM_PROMPT)
    return make_alpaca(inp, out) if out else None


def gen_portfolio_analyze() -> dict:
    p = sample_persona()
    all_funds = MF_CATS["equity"] + MF_CATS["hybrid"] + MF_CATS["passive"] + MF_CATS["debt"]
    selected  = random.sample(all_funds, random.randint(3, 8))
    total     = random.choice([50000, 100000, 200000, 500000, 1000000, 2000000])

    remaining = 100
    allocs = []
    for i, fund in enumerate(selected):
        if i == len(selected) - 1:
            pct = remaining
        else:
            pct = random.randint(5, min(40, remaining - (len(selected)-i-1)*5))
            remaining -= pct
        allocs.append((fund, pct))

    pf_str = "\n".join(f"- {f}: {pct}% (₹{total*pct//100:,})" for f, pct in allocs)
    qs = [
        "Is my portfolio well-diversified?",
        "Is my portfolio too aggressive for my age?",
        "Should I consolidate my funds?",
        "I've been investing 3 years — review my portfolio and suggest changes.",
    ]
    inp = f"{random.choice(qs)}\n\nPortfolio (Total ₹{total:,}):\n{pf_str}\n\nAge: {p['age']}, risk: {p['risk'].replace('_',' ')}."

    prompt = f"""Analyze this Indian investor's mutual fund portfolio.

{pf_str}
Total: ₹{total:,} | Age: {p['age']} | Risk: {p['risk'].replace('_',' ')}

Provide:
1. Diversification assessment
2. Category overlap analysis
3. Age-appropriate allocation check
4. Issues identified
5. Actionable recommendations
6. Suggested target allocation
7. Risk disclaimer

Markdown. 250-500 words. Output ONLY the analysis."""
    out = call_llm(prompt, system=SYSTEM_PROMPT)
    return make_alpaca(inp, out) if out else None


def gen_compare() -> dict:
    a, b = random.choice(COMPARISON_PAIRS)
    p = sample_persona()
    inp = random.choice([
        f"What is better — {a} or {b}? I am {p['age']} years old, ₹{p['invest']:,}/month to invest.",
        f"Compare {a} vs {b} for an Indian investor.",
        f"Help me decide between {a} and {b}. My goal is {p['goal'].replace('_',' ')}.",
        f"I'm confused between {a} and {b}. Which should I choose?",
    ])
    prompt = f"""Compare these two Indian financial instruments comprehensively.

A: {a}
B: {b}
Investor: Age {p['age']}, ₹{p['invest']:,}/month, goal: {p['goal'].replace('_',' ')}, risk: {p['risk'].replace('_',' ')}

Include:
1. One-line overview of each
2. Comparison table: Returns | Risk | Liquidity | Taxation | Lock-in | Min investment | Best for
3. Indian tax implications for each
4. Clear "when to choose which" guidance
5. Risk disclaimer

Markdown table required. 200-450 words. Output ONLY the comparison."""
    out = call_llm(prompt, system=SYSTEM_PROMPT)
    return make_alpaca(inp, out) if out else None


def gen_tax() -> dict:
    scenario, hint = random.choice(TAX_SCENARIOS)
    prompt = f"""Generate an accurate Indian taxation guidance response.

User question: "{scenario}"
Key concept to cover: {hint}

Requirements:
- Accurate to Indian Income Tax Act FY2024-25
- Specific numbers (rates, limits, slabs)
- Practical — what should the person actually do?
- Suggest CA consultation for complex situations
- Disclaimer at end

150-350 words, markdown. Output ONLY the response."""
    out = call_llm(prompt, system=SYSTEM_PROMPT)
    return make_alpaca(scenario, out) if out else None


def gen_refusal() -> dict:
    scenario = random.choice(REFUSAL_SCENARIOS)
    prompt = f"""Generate a polite but firm refusal for an Indian financial advisor chatbot.

User request: "{scenario}"

This involves: insider trading / guaranteed returns / illegal activity / pump-and-dump / price prediction.

Response must:
1. Decline clearly but kindly
2. Explain WHY (legal risk, SEBI regulation, or market reality)
3. Offer a constructive alternative
4. Stay helpful and non-judgmental

100-250 words, conversational tone. Output ONLY the response."""
    out = call_llm(prompt, system=SYSTEM_PROMPT)
    return make_alpaca(scenario, out) if out else None


# ── Router ────────────────────────────────────────────────────────────────────

GENERATORS = {
    "qa":               gen_qa,
    "stock":            gen_stock,
    "mf":               gen_mf,
    "portfolio_build":  gen_portfolio_build,
    "portfolio_analyze":gen_portfolio_analyze,
    "compare":          gen_compare,
    "tax":              gen_tax,
    "refusal":          gen_refusal,
}

WEIGHTS = {
    "qa": 0.15, "stock": 0.15, "mf": 0.15,
    "portfolio_build": 0.20, "portfolio_analyze": 0.10,
    "compare": 0.10, "tax": 0.08, "refusal": 0.07,
}

BANNED = ["guaranteed returns","will definitely","100% safe",
          "no risk","assured profit","sure shot","100% guarantee"]

def validate(s: dict) -> tuple:
    if not s:
        return False, "None"
    words = len(s["output"].split())
    if words < 80:  return False, f"Too short ({words}w)"
    if words > 900: return False, f"Too long ({words}w)"
    low = s["output"].lower()
    for p in BANNED:
        if p in low: return False, f"Banned: '{p}'"
    return True, "ok"

def dhash(s: dict) -> str:
    t = s.get("input","") + s.get("output","")[:200]
    return hashlib.md5(t.encode()).hexdigest()

# ── Main ──────────────────────────────────────────────────────────────────────

def run(total: int, task: str, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    generated, rejected, seen, samples = 0, 0, set(), []

    print(f"\n🚀 Generating {total} samples  task={task}  →  {out_path}\n" + "-"*50)

    while generated < total:
        t = task if task != "all" else random.choices(list(WEIGHTS), weights=list(WEIGHTS.values()))[0]
        try:
            s = GENERATORS[t]()
        except Exception as e:
            print(f"  ❌ {t}: {e}"); rejected += 1; continue

        ok, reason = validate(s)
        if not ok:
            print(f"  ⚠️  [{t}] {reason}"); rejected += 1; continue

        h = dhash(s)
        if h in seen:
            rejected += 1; continue
        seen.add(h)

        s["_task"] = t
        s["_id"]   = generated + 1
        samples.append(s)
        generated += 1
        print(f"  ✅ [{generated}/{total}] {t}")

        if generated % 50 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                for x in samples: f.write(json.dumps(x, ensure_ascii=False) + "\n")

        time.sleep(0.4)

    with open(out_path, "w", encoding="utf-8") as f:
        for x in samples: f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"✅ Done  generated={generated}  rejected={rejected}  "
          f"reject_rate={rejected/(generated+rejected)*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",   default="all", choices=["all"]+list(GENERATORS))
    ap.add_argument("--count",  type=int, default=100)
    ap.add_argument("--output", default="../dataset/raw/samples.jsonl")
    args = ap.parse_args()
    run(args.count, args.task, args.output)
