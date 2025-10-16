#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple, Optional
import argparse
import sys
from collections import Counter

FALLBACK_WORDS = "able bale ball bell label belly cable call cell clay ally lay lab lace".split()

@dataclass
class BeeConfig:
    required: str
    optional: Set[str]
    min_len: int = 4
    allow_proper_nouns: bool = False
    @property
    def alphabet(self) -> Set[str]:
        return {self.required, *self.optional}

def load_dictionary(path: Optional[str], n_top: int, min_zipf: Optional[float]) -> List[str]:
    if path:
        words: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    words.append(w)
        print(f"[INFO] Loaded {len(words)} words from file: {path}")
        return words

    try:
        from wordfreq import top_n_list, zipf_frequency
    except ImportError:
        words = [w.lower() for w in FALLBACK_WORDS]
        print(f"[WARN] wordfreq not installed. Using FALLBACK ({len(words)} words).")
        return words

    # NOTE: `n_top` is positional, not keyword, in your version
    words = [w for w in top_n_list("en", n_top) if w.isalpha()]
    if min_zipf is not None:
        words = [w for w in words if zipf_frequency(w, "en") >= min_zipf]
    print(f"[INFO] Loaded {len(words)} words from wordfreq (n_top={n_top}, min_zipf={min_zipf}).")
    return words

def is_valid_word(w: str, cfg: BeeConfig) -> Tuple[bool, bool]:
    if len(w) < cfg.min_len:
        return (False, False)
    if not cfg.allow_proper_nouns and (w[0].isupper() or not w.islower()):
        return (False, False)
    if not w.isalpha():
        return (False, False)
    s = set(w)
    if cfg.required not in s:
        return (False, False)
    if not s.issubset(cfg.alphabet):
        return (False, False)
    is_pangram = cfg.alphabet.issubset(s)
    return (True, is_pangram)

def score_word(w: str, is_pangram: bool, base_min_len: int) -> int:
    pts = 1 if len(w) == base_min_len else len(w)
    if is_pangram:
        pts += 7
    return pts

def find_words(cfg: BeeConfig, words: Iterable[str], debug: bool = False) -> List[tuple]:
    if not debug:
        out = []
        for w in words:
            valid, pang = is_valid_word(w, cfg)
            if valid:
                out.append((w, pang, score_word(w, pang, cfg.min_len)))
        out.sort(key=lambda t: (not t[1], -t[2], t[0]))
        return out

    # Debug path: track where rejections happen
    stages = Counter()
    survivors = []  # final
    samples = {  # keep up to 10 examples per stage
        "too_short": [], "not_lower_or_alpha": [], "missing_required": [],
        "uses_outside_letters": [], "ok": []
    }

    for w in words:
        if len(w) < cfg.min_len:
            stages["too_short"] += 1
            if len(samples["too_short"]) < 10: samples["too_short"].append(w)
            continue

        if not w.isalpha() or (not cfg.allow_proper_nouns and (w[0].isupper() or not w.islower())):
            stages["not_lower_or_alpha"] += 1
            if len(samples["not_lower_or_alpha"]) < 10: samples["not_lower_or_alpha"].append(w)
            continue

        s = set(w)
        if cfg.required not in s:
            stages["missing_required"] += 1
            if len(samples["missing_required"]) < 10: samples["missing_required"].append(w)
            continue

        if not s.issubset(cfg.alphabet):
            stages["uses_outside_letters"] += 1
            if len(samples["uses_outside_letters"]) < 10: samples["uses_outside_letters"].append(w)
            continue

        pang = cfg.alphabet.issubset(s)
        survivors.append((w, pang, score_word(w, pang, cfg.min_len)))
        stages["ok"] += 1
        if len(samples["ok"]) < 10: samples["ok"].append(w)

    survivors.sort(key=lambda t: (not t[1], -t[2], t[0]))

    # Print debug summary
    print("DEBUG summary:")
    total = sum(stages.values())
    print(f"  Total considered: {total}")
    for k in ["too_short", "not_lower_or_alpha", "missing_required", "uses_outside_letters", "ok"]:
        print(f"  {k:22s}: {stages[k]}")
        if samples[k]:
            print(f"    e.g.: {', '.join(samples[k])}")

    return survivors

def parse_letters(required: str, optional: str) -> Tuple[str, Set[str]]:
    required = required.lower()
    optional = optional.lower()
    if len(required) != 1 or not required.isalpha():
        raise ValueError("Required must be a single letter A-Z.")
    if len(optional) != 6 or not optional.isalpha():
        raise ValueError("Optional must be exactly 5 letters A-Z.")
    opt_set = set(optional)
    if required in opt_set:
        raise ValueError("Required letter must not be among the 5 optional letters.")
    if len(opt_set) != 6:
        raise ValueError("Optional letters must be 5 distinct letters.")
    return required, opt_set

def main(argv=None):
    p = argparse.ArgumentParser(description="Spelling Bee word finder with wordfreq integration.")
    p.add_argument("-r", "--required", required=True, help="One required letter.")
    p.add_argument("-o", "--optional", required=True, help="Five optional letters, e.g. bcdef.")
    p.add_argument("-d", "--dict", default=None, help="Path to a word list file (one word per line). If omitted, uses wordfreq if available.")
    p.add_argument("-m", "--min-len", type=int, default=4, help="Minimum word length (default 4).")
    p.add_argument("--allow-proper-nouns", action="store_true", help="Allow capitalized words.")
    # wordfreq knobs:
    p.add_argument("--n-top", type=int, default=200000, help="When using wordfreq, size of list to load (default 200000).")
    p.add_argument("--min-zipf", type=float, default=3.5, help="Keep words with Zipf freq >= this (default 3.5). Use 0 to disable.")
    p.add_argument("--debug", action="store_true", help="Print filter-stage stats and samples.")
    args = p.parse_args(argv)

    req, opt = parse_letters(args.required, args.optional)
    cfg = BeeConfig(required=req, optional=opt, min_len=args.min_len, allow_proper_nouns=args.allow_proper_nouns)

    min_zipf = None if args.min_zipf and args.min_zipf <= 0 else args.min_zipf
    words = load_dictionary(args.dict, n_top=args.n_top, min_zipf=min_zipf)
    results = find_words(cfg, words, debug=args.debug)

    if not results:
        print("No matches found. Try lowering --min-zipf, increasing --n-top, or using a bigger dictionary.")
        sys.exit(0)

    total_score = sum(s for _, _, s in results)
    pangrams = [w for (w, pang, _) in results if pang]

    print(f"Required: {cfg.required} | Optional: {''.join(sorted(cfg.optional))} | Min length: {cfg.min_len}")
    print(f"Alphabet: {''.join(sorted(cfg.alphabet))}")
    print(f"Loaded {len(words)} words. Found {len(results)} matches | Pangrams: {len(pangrams)} | Total score: {total_score}\n")

    if pangrams:
        print("Pangrams:")
        for w, pang, s in results:
            if pang:
                print(f"  {w} (+7) [{s}]")
        print()

    print("All matches:")
    for w, pang, s in results:
        flag = " [PANGRAM]" if pang else ""
        print(f"  {w}{flag} [{s}]")

if __name__ == "__main__":
    main()
