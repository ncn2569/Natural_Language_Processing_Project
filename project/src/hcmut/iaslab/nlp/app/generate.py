"""
->> Nên dùng generator.py trong src/grammar/
"""
# import itertools
# import os
# import random
# import re
# import shutil
# import sys
# from pathlib import Path

# EPSILON = "ε"
# MAX_SENTENCES = 10000
# # .../python/hcmut/iaslab/nlp/app/generate.py -> .../python
# PROJECT_ROOT = Path(__file__).resolve().parents[4]


# def default_dirs():
#     """Docker cham bai mount /nlp/input va /nlp/output; chay local thi dung thu muc trong project."""
#     in_dir = os.environ.get("NLP_INPUT") or ("/nlp/input" if os.path.isdir("/nlp/input") else str(PROJECT_ROOT.parent / "input"))
#     out_dir = os.environ.get("NLP_OUTPUT") or ("/nlp/output" if os.path.isdir("/nlp/output") else str(PROJECT_ROOT.parent / "output"))
#     return in_dir, out_dir


# def load_grammar(path):
#     rules, start = {}, None
#     with open(path, encoding="utf-8") as f:
#         for ln, raw in enumerate(f, 1):
#             line = raw.strip()
#             if not line or line.startswith("#"):
#                 continue
#             if "->" not in line:
#                 raise ValueError(f"{path}:{ln}: thieu '->': {line}")
#             lhs, rhs = (s.strip() for s in line.split("->", 1))
#             start = start or lhs
#             for alt in rhs.split("|"):
#                 toks = alt.split()
#                 if not toks:
#                     raise ValueError(f"{path}:{ln}: phuong an rong cho {lhs} (dung {EPSILON})")
#                 rules.setdefault(lhs, []).append(toks)
#     if start is None:
#         raise ValueError("Van pham rong")
#     return rules, start


# def check_grammar(rules):
#     visiting, done = set(), set()

#     def dfs(nt):
#         if nt in done:
#             return
#         if nt in visiting:
#             raise ValueError(f"Van pham de quy vong quanh '{nt}' -> khong sinh huu han duoc")
#         visiting.add(nt)
#         for alt in rules[nt]:
#             for t in alt:
#                 if t in rules:
#                     dfs(t)
#         visiting.discard(nt)
#         done.add(nt)

#     for nt in rules:
#         dfs(nt)


# class Generator:
#     def __init__(self, rules):
#         self.rules = rules
#         self._count = {}

#     def count(self, nt):
#         if nt not in self._count:
#             self._count[nt] = sum(self.count_alt(a) for a in self.rules[nt])
#         return self._count[nt]

#     def count_alt(self, alt):
#         c = 1
#         for t in alt:
#             if t in self.rules:
#                 c *= self.count(t)
#         return c

#     def enumerate(self, symbol):
#         if symbol not in self.rules:
#             yield [] if symbol == EPSILON else [symbol]
#             return
#         for alt in self.rules[symbol]:
#             lists = [list(self.enumerate(t)) for t in alt]
#             for combo in itertools.product(*lists):
#                 yield [w for seg in combo for w in seg]

#     def sample(self, symbol, rng):
#         if symbol not in self.rules:
#             return [] if symbol == EPSILON else [symbol]
#         alts = self.rules[symbol]
#         alt = rng.choices(alts, weights=[self.count_alt(a) for a in alts], k=1)[0]
#         out = []
#         for t in alt:
#             out.extend(self.sample(t, rng))
#         return out


# def normalize(words):
#     s = re.sub(r"\s+", " ", " ".join(words)).strip()
#     s = re.sub(r"\s+\?", "?", s)
#     return s[:1].upper() + s[1:] if s else s


# def generate_sentences(rules, start, limit=MAX_SENTENCES, seed=42):
#     gen = Generator(rules)
#     total = gen.count(start)
#     sentences, seen = [], set()
#     if total <= limit:
#         for w in gen.enumerate(start):
#             s = normalize(w)
#             if s not in seen:
#                 seen.add(s)
#                 sentences.append(s)
#         mode = "enumerate"
#     else:
#         rng, attempts = random.Random(seed), 0
#         while len(sentences) < limit and attempts < limit * 50:
#             attempts += 1
#             s = normalize(gen.sample(start, rng))
#             if s not in seen:
#                 seen.add(s)
#                 sentences.append(s)
#         sentences.sort()
#         mode = "sample"
#     return sentences, total, mode


# def run(**kwargs):
#     """Entry point cho hcommon.cli (tham so lay tu conf/app.json) hoac goi truc tiep."""
#     _, out_dir = default_dirs()
#     out_dir = kwargs.get("output_dir", out_dir)
#     grammar = kwargs.get("grammar", str(PROJECT_ROOT / "data" / "grammar.txt"))
#     limit = min(int(kwargs.get("max_sentences", MAX_SENTENCES)), MAX_SENTENCES)
#     seed = int(kwargs.get("seed", 42))

#     rules, start = load_grammar(grammar)
#     check_grammar(rules)
#     sentences, total, mode = generate_sentences(rules, start, limit, seed)

#     os.makedirs(out_dir, exist_ok=True)
#     shutil.copyfile(grammar, os.path.join(out_dir, "grammar.txt"))          # 3.1
#     with open(os.path.join(out_dir, "samples.txt"), "w", encoding="utf-8") as f:  # 3.2
#         f.write("\n".join(sentences) + "\n")
#     print(f"[3.1] grammar -> {out_dir}/grammar.txt")
#     print(f"[3.2] {len(sentences)} cau ({mode}, tong so dan xuat={total}) -> {out_dir}/samples.txt")


# if __name__ == "__main__":
#     run()
