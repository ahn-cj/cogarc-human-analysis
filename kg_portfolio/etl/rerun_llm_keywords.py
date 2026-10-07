"""Re-run the keyword task against current models.

The GPT-4o / Gemini descriptions in this repo are from 2024 and predate the
reasoning-model generation. This repeats the task with any model you point it
at and writes data/<model>_keywords.csv, which etl/add_agents.py picks up
automatically.

Two input modes, because they are different experiments:

  --mode image   (default) sends the SAME rendered PNGs the participants saw.
                 This is the faithful comparison: the human task was visual.
  --mode text    sends the grids as digit matrices. Easier, and what the 2024
                 run did, but it is not the task a person was given.

Providers. One key gets you most things through OpenRouter, and the `openai`
package is already installed, so that is the low-friction path:

  export OPENROUTER_API_KEY=...
  python etl/rerun_llm_keywords.py --provider openrouter \
      --models google/gemini-3-pro meta-llama/llama-4-maverick

Native SDKs also work if you prefer:

  export GEMINI_API_KEY=...   # pip install google-genai
  python etl/rerun_llm_keywords.py --provider gemini --models gemini-3-pro

  export TOGETHER_API_KEY=... # Llama, OpenAI-compatible endpoint
  python etl/rerun_llm_keywords.py --provider together \
      --models meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8

Responses are cached per (model, mode, puzzle), so re-running is free for
puzzles already done and you can stop and resume.
"""
from __future__ import annotations
import argparse, base64, glob, json, os, re, sys, time

import paths
KWT = str(paths.KEYWORD_TESTSET)
PNG = str(paths.KEYWORD_PNG)
CACHE, OUT = str(paths.DATA / "llm_cache"), str(paths.DATA)

VOCAB = ["Color", "Contact", "Direction", "Distance", "Fill", "Number", "Object",
         "Order", "Path", "Pattern", "Position", "Reverse", "Rotation", "Shape",
         "Size", "Symmetry"]

INSTRUCTION = """You are shown an abstract reasoning puzzle. Each example is an input \
grid and the output grid it transforms into. Work out the rule that turns each input \
into its output.

Below is a fixed list of sixteen words. Choose up to five that are relevant to the rule, \
ordered from most to least central to it. Use only words from the list. Reply with the \
chosen words on one line, comma separated, and nothing else.

{vocab}"""

ENDPOINTS = {                     # OpenAI-compatible bases
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together":   ("https://api.together.xyz/v1",  "TOGETHER_API_KEY"),
    "groq":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai":     (None,                           "OPENAI_API_KEY"),
}


def text_examples(idx: str) -> str:
    j = json.load(open(f"{KWT}/{idx}.json"))
    out = []
    for n, ex in enumerate(j["train"], 1):
        gi = "\n".join("".join(str(c) for c in r) for r in ex["input"])
        go = "\n".join("".join(str(c) for c in r) for r in ex["output"])
        out.append(f"Example {n} input:\n{gi}\n\nExample {n} output:\n{go}")
    return "\n\n".join(out)


def image_parts(idx: str) -> list[tuple[str, str]]:
    """The participant's own stimuli: (label, path) for each train pair, in order."""
    out = []
    for n in range(10):
        i, o = f"{PNG}/{idx}_train_{n}_input.png", f"{PNG}/{idx}_train_{n}_output.png"
        if not (os.path.exists(i) and os.path.exists(o)):
            break
        out += [(f"Example {n+1} input", i), (f"Example {n+1} output", o)]
    if not out:
        raise FileNotFoundError(f"no rendered images for puzzle {idx} in {PNG}")
    return out


def b64(path: str) -> str:
    return base64.standard_b64encode(open(path, "rb").read()).decode()


def build_oai_content(idx: str, mode: str, vocab: str):
    if mode == "text":
        return f"{text_examples(idx)}\n\n{INSTRUCTION.format(vocab=vocab)}"
    parts = []
    for label, path in image_parts(idx):
        parts.append({"type": "text", "text": label})
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/png;base64,{b64(path)}"}})
    parts.append({"type": "text", "text": INSTRUCTION.format(vocab=vocab)})
    return parts


def call(provider: str, model: str, idx: str, mode: str, vocab: str) -> str:
    if provider == "gemini":
        from google import genai
        from google.genai import types
        c = genai.Client()
        if mode == "text":
            contents = [f"{text_examples(idx)}\n\n{INSTRUCTION.format(vocab=vocab)}"]
        else:
            contents = []
            for label, path in image_parts(idx):
                contents.append(label)
                contents.append(types.Part.from_bytes(
                    data=open(path, "rb").read(), mime_type="image/png"))
            contents.append(INSTRUCTION.format(vocab=vocab))
        return c.models.generate_content(model=model, contents=contents).text

    base, env = ENDPOINTS[provider]
    if not os.environ.get(env):
        raise SystemExit(f"{env} is not set")
    from openai import OpenAI
    c = OpenAI(api_key=os.environ[env], **({"base_url": base} if base else {}))
    r = c.chat.completions.create(
        model=model, max_tokens=200,
        messages=[{"role": "user", "content": build_oai_content(idx, mode, vocab)}])
    return r.choices[0].message.content


def parse(text: str) -> list[str]:
    """Keep vocabulary words in the order given, deduplicated, max five."""
    found, seen = [], set()
    for tok in re.split(r"[,\n]", text or ""):
        w = tok.strip().strip(".-*`0123456789 ").title()
        if w in VOCAB and w not in seen:
            seen.add(w); found.append(w)
    return found[:5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--provider", default="openrouter", choices=list(ENDPOINTS) + ["gemini"])
    ap.add_argument("--mode", default="image", choices=["image", "text"])
    ap.add_argument("--puzzles", nargs="*", default=[f"{i:02d}" for i in range(1, 41)])
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    vocab = ", ".join(VOCAB)

    for model in a.models:
        slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
        rows, misses = [], 0
        for idx in a.puzzles:
            cf = os.path.join(CACHE, f"{slug}_{a.mode}_{idx}.txt")
            if os.path.exists(cf):
                txt = open(cf).read()
            else:
                try:
                    txt = call(a.provider, model, idx, a.mode, vocab)
                except Exception as e:
                    print(f"  {model} {idx}: {e}", file=sys.stderr); misses += 1; continue
                open(cf, "w").write(txt or ""); time.sleep(0.4)
            kws = parse(txt)
            if not kws: misses += 1
            rows.append({"task": idx, **{f"ai_keyword{i+1}": (kws[i] if i < len(kws) else "")
                                          for i in range(5)}})
            print(f"  {model} {idx}: {', '.join(kws) or '(nothing parsed)'}")
        import pandas as pd
        p = os.path.join(OUT, f"{slug}_keywords.csv")
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f"wrote {os.path.normpath(p)}  {len(rows)} puzzles, {misses} without a usable "
              f"answer\n  then: python etl/add_agents.py && python etl/export_site.py "
              f"&& python etl/build_site.py\n")


if __name__ == "__main__":
    main()
